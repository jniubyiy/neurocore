// src/layers/relative_position_attention/cpu/mod.rs

use crate::compute_manager::graph::types::DynamicContext;
use crate::compute_manager::matrix_buffer::MatrixBufferHandle;
use crate::layers::buffered_context::BufferedContext;
use crate::layers::UniversalLayerBuffered;
use crate::model_plan::param_store::ParamSlice;

use super::super::relative_position_attention::RelativePositionAttention;

impl UniversalLayerBuffered for RelativePositionAttention {
    fn forward_buffered(
        &self,
        input: &MatrixBufferHandle,
        output: &MatrixBufferHandle,
        params: &MatrixBufferHandle,
        slice: &ParamSlice,
    ) {
        let batch = input.rows();
        let features = input.cols();
        let d = self.d_model;
        let seq = self.seq_len;
        debug_assert_eq!(features, seq * d);
        debug_assert!(slice.start + self.param_len() <= params.rows() * params.cols());

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
            let bias_start = bo_start + d; // относительное смещение (2*seq-1)

            // Вспомогательные буферы
            let mut q = vec![0.0f32; batch * seq * d];
            let mut k = vec![0.0f32; batch * seq * d];
            let mut v = vec![0.0f32; batch * seq * d];

            // Вычисляем Q, K, V
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
                        let q_idx = (r * seq + t) * d + j;
                        q[q_idx] = sum_q;
                        k[q_idx] = sum_k;
                        v[q_idx] = sum_v;
                    }
                }
            }

            // Вычисляем внимание для каждого батча и каждой пары позиций
            let mut attn_out = vec![0.0f32; batch * seq * d];
            for r in 0..batch {
                for t in 0..seq {
                    // Для каждого запроса t вычисляем softmax по s
                    let mut scores = vec![0.0f32; seq];
                    let mut max_score = f32::NEG_INFINITY;
                    for s in 0..seq {
                        let mut score = 0.0;
                        for j in 0..d {
                            score += q[(r * seq + t) * d + j] * k[(r * seq + s) * d + j];
                        }
                        score /= (d as f32).sqrt();
                        // добавляем относительное смещение
                        let rel_idx = (s as isize - t as isize + (seq as isize - 1)) as usize;
                        score += p[bias_start + rel_idx];
                        scores[s] = score;
                        if score > max_score { max_score = score; }
                    }
                    // softmax
                    let mut sum_exp = 0.0;
                    let mut exps = vec![0.0f32; seq];
                    for s in 0..seq {
                        let e = (scores[s] - max_score).exp();
                        exps[s] = e;
                        sum_exp += e;
                    }
                    // взвешенная сумма V
                    for j in 0..d {
                        let mut out_val = 0.0;
                        for s in 0..seq {
                            let w = exps[s] / sum_exp;
                            out_val += w * v[(r * seq + s) * d + j];
                        }
                        attn_out[(r * seq + t) * d + j] = out_val;
                    }
                }
            }

            // Выходное линейное преобразование
            for r in 0..batch {
                for t in 0..seq {
                    for j in 0..d {
                        let mut sum = p[bo_start + j];
                        for i in 0..d {
                            sum += attn_out[(r * seq + t) * d + i] * p[wo_start + j * d + i];
                        }
                        let out_idx = (t * d + j) * batch + r;
                        y[out_idx] = sum;
                    }
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
        // Полная реализация обратного прохода очень объёмна.
        // Для текущей версии не поддерживается.
        panic!("RelativePositionAttention backward is not implemented yet");
    }

    fn param_len(&self) -> usize {
        let d = self.d_model;
        4 * (d * d + d) + (2 * self.seq_len - 1)
    }

    fn input_features(&self) -> usize {
        self.seq_len * self.d_model
    }

    fn output_features(&self) -> usize {
        self.seq_len * self.d_model
    }
}