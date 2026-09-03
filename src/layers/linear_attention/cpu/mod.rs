// src/layers/linear_attention/cpu/mod.rs

use crate::compute_manager::graph::types::DynamicContext;
use crate::compute_manager::matrix_buffer::MatrixBufferHandle;
use crate::layers::buffered_context::BufferedContext;
use crate::layers::UniversalLayerBuffered;
use crate::model_plan::param_store::ParamSlice;

use super::super::linear_attention::LinearAttention;

impl UniversalLayerBuffered for LinearAttention {
    fn forward_buffered(
        &self,
        input: &MatrixBufferHandle,
        output: &MatrixBufferHandle,
        params: &MatrixBufferHandle,
        slice: &ParamSlice,
    ) {
        let batch = input.rows();
        let features = input.cols();
        debug_assert_eq!(features, self.seq_len * self.d_model);
        debug_assert!(slice.start + self.param_len() <= params.rows() * params.cols());

        let d = self.d_model;
        let seq = self.seq_len;

        // Читаем входные данные, параметры и выход
        let ids = [input.id(), output.id(), params.id()];
        input.memory().write().unwrap().with_cpu_slices_mut(&ids, |slices| {
            let (first, rest) = slices.split_at_mut(1);
            let x: &[f32] = &*first[0];
            let (second, rest) = rest.split_at_mut(1);
            let y: &mut [f32] = &mut *second[0];
            let p: &[f32] = &*rest[0];

            let base = slice.start;

            // Смещения параметров
            let wq_start = base;
            let bq_start = wq_start + d * d;
            let wk_start = bq_start + d;
            let bk_start = wk_start + d * d;
            let wv_start = bk_start + d;
            let bv_start = wv_start + d * d;
            let wo_start = bv_start + d;
            let bo_start = wo_start + d * d;

            // Временные буферы для Q, K, V (batch × d)
            let mut q = vec![0.0f32; batch * d];
            let mut k = vec![0.0f32; batch * d];
            let mut v = vec![0.0f32; batch * d];

            // Линейные преобразования Q = X W_q + b_q, K = X W_k + b_k, V = X W_v + b_v
            // X размер (batch × seq*d), мы будем трактовать X как (batch, seq, d) в row-major по последнему измерению.
            // В column-major это не очень удобно, поэтому преобразуем в row-major вспомогательный массив?
            // Проще: для каждой головы (одна) и каждого батча и позиции мы вычисляем вручную.
            // Вход x в column-major: x[c * batch + r], где c - признак (0..features-1).
            // Признак c = t * d + j, где t - позиция, j - компонента d_model.
            // Тогда x[(t*d + j) * batch + r] = X[r, t, j].

            // Вычисляем Q, K, V для каждого батча и позиции.
            for r in 0..batch {
                for t in 0..seq {
                    for j in 0..d {
                        let mut sum_q = p[bq_start + j];
                        let mut sum_k = p[bk_start + j];
                        let mut sum_v = p[bv_start + j];
                        for i in 0..d {
                            let x_idx = (t * d + i) * batch + r;
                            let x_val = x[x_idx];
                            sum_q += x_val * p[wq_start + j * d + i];
                            sum_k += x_val * p[wk_start + j * d + i];
                            sum_v += x_val * p[wv_start + j * d + i];
                        }
                        q[(t * batch + r) * d + j] = sum_q;
                        k[(t * batch + r) * d + j] = sum_k;
                        v[(t * batch + r) * d + j] = sum_v;
                    }
                }
            }

            // Применяем phi(x) = ELU(x) + 1
            for i in 0..q.len() {
                q[i] = elu(q[i]) + 1.0;
                k[i] = elu(k[i]) + 1.0;
            }

            // Вычисляем KV = K^T V, размер d × d
            // K имеет размер (batch*seq, d), V (batch*seq, d)
            // K^T V = sum_r K[r, :]^T V[r, :]
            let mut kv = vec![0.0f32; d * d]; // row-major: kv[i*d + j]
            for r in 0..(batch * seq) {
                let k_row = &k[r * d..(r + 1) * d];
                let v_row = &v[r * d..(r + 1) * d];
                for i in 0..d {
                    let ki = k_row[i];
                    for j in 0..d {
                        kv[i * d + j] += ki * v_row[j];
                    }
                }
            }

            // Вычисляем Z = K^T 1, размер d
            let mut z = vec![0.0f32; d];
            for r in 0..(batch * seq) {
                let k_row = &k[r * d..(r + 1) * d];
                for i in 0..d {
                    z[i] += k_row[i];
                }
            }

            // Вычисляем выход до линейного преобразования: Y = (Q * KV) / (Q * Z + eps)
            // Q размер (batch*seq, d), результат (batch*seq, d)
            let eps = 1e-6;
            let mut attn_out = vec![0.0f32; batch * seq * d];
            for r in 0..(batch * seq) {
                let q_row = &q[r * d..(r + 1) * d];
                // Вычисляем denom = sum_i q_i * z_i
                let mut denom = 0.0f32;
                for i in 0..d {
                    denom += q_row[i] * z[i];
                }
                denom += eps;

                for j in 0..d {
                    // numerator = sum_i q_i * kv[i, j]
                    let mut num = 0.0f32;
                    for i in 0..d {
                        num += q_row[i] * kv[i * d + j];
                    }
                    attn_out[r * d + j] = num / denom;
                }
            }

            // Линейное преобразование выхода: Y_final = attn_out W_o + b_o
            for r in 0..(batch * seq) {
                for j in 0..d {
                    let mut sum = p[bo_start + j];
                    for i in 0..d {
                        sum += attn_out[r * d + i] * p[wo_start + j * d + i];
                    }
                    // Записываем в выход, используя column-major: признак (t*d+j) для батча r
                    let t = r % seq;
                    let b = r / seq;
                    let out_idx = (t * d + j) * batch + b;
                    y[out_idx] = sum;
                }
            }
        });
    }

    fn backward_buffered(
        &self,
        _ctx: &DynamicContext,
        _grad_output: &MatrixBufferHandle,
        _grad_input: &MatrixBufferHandle,
        _params: &MatrixBufferHandle,
        _slice: &ParamSlice,
        _grad_params: &MatrixBufferHandle,
    ) {
        // Полная реализация обратного прохода требует значительного объёма кода.
        // Для текущей версии обратный проход не поддерживается.
        panic!("LinearAttention backward is not implemented yet");
    }

    fn param_len(&self) -> usize {
        let d = self.d_model;
        4 * (d * d + d)
    }

    fn input_features(&self) -> usize {
        self.seq_len * self.d_model
    }

    fn output_features(&self) -> usize {
        self.seq_len * self.d_model
    }
}

fn elu(x: f32) -> f32 {
    if x > 0.0 { x } else { x.exp() - 1.0 }
}