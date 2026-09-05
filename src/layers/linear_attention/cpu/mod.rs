// src/layers/linear_attention/cpu/mod.rs

use crate::compute_manager::graph::types::DynamicContext;
use crate::compute_manager::matrix_buffer::MatrixBufferHandle;
use crate::layers::buffered_context::BufferedContext;
use crate::layers::UniversalLayerBuffered;
use crate::model_plan::param_store::ParamSlice;

use super::super::linear_attention::{LinearAttention, LinearAttentionCache};

// ====================== Вспомогательные функции ======================

/// Вычисляет phi(x) = ELU(x) + 1 (поэлементно).
fn phi(x: f32) -> f32 {
    if x > 0.0 { x + 1.0 } else { x.exp() + 1.0 }
}

/// Производная phi(x) по x (для обратного прохода).
fn phi_derivative(x: f32) -> f32 {
    if x > 0.0 { 1.0 } else { x.exp() }
}

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
        let d = self.d_model;
        let seq = self.seq_len;
        let total_tokens = seq * d;

        debug_assert_eq!(features, total_tokens);
        debug_assert_eq!(output.rows(), batch);
        debug_assert_eq!(output.cols(), total_tokens);
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

            // Временные буферы
            let tokens_per_batch = seq * d; // общее число элементов на один пример (признаков)
            let mut q_raw = vec![0.0f32; batch * tokens_per_batch];
            let mut k_raw = vec![0.0f32; batch * tokens_per_batch];
            let mut v_raw = vec![0.0f32; batch * tokens_per_batch];

            // Преобразуем входной column-major в row-major для удобства внутренних вычислений.
            // Входной x: признаки = (t, j), column-major: x[(t*d + j)*batch + r]
            // Мы можем оставить column-major, но для матричных операций удобнее row-major.
            // Поэтому переупорядочим: для каждого примера r создадим матрицу (seq x d) row-major.
            let mut x_rows = vec![0.0f32; batch * tokens_per_batch]; // row-major: r * tokens + (t*d + j)
            for r in 0..batch {
                for t in 0..seq {
                    for j in 0..d {
                        let src_idx = (t * d + j) * batch + r;
                        let dst_idx = r * tokens_per_batch + t * d + j;
                        x_rows[dst_idx] = x[src_idx];
                    }
                }
            }

            // Линейные преобразования
            for r in 0..batch {
                for t in 0..seq {
                    let offset = r * tokens_per_batch + t * d;
                    for j in 0..d {
                        // Q
                        let mut sum = p[bq_start + j];
                        for i in 0..d {
                            sum += x_rows[offset + i] * p[wq_start + j * d + i];
                        }
                        q_raw[offset + j] = sum;

                        // K
                        sum = p[bk_start + j];
                        for i in 0..d {
                            sum += x_rows[offset + i] * p[wk_start + j * d + i];
                        }
                        k_raw[offset + j] = sum;

                        // V
                        sum = p[bv_start + j];
                        for i in 0..d {
                            sum += x_rows[offset + i] * p[wv_start + j * d + i];
                        }
                        v_raw[offset + j] = sum;
                    }
                }
            }

            // Применяем phi к Q и K
            let mut q_phi = vec![0.0f32; batch * tokens_per_batch];
            let mut k_phi = vec![0.0f32; batch * tokens_per_batch];
            for i in 0..q_raw.len() {
                q_phi[i] = phi(q_raw[i]);
                k_phi[i] = phi(k_raw[i]);
            }
            // V остаётся без phi

            // Вычисляем глобальные суммы KV и Z (по всем примерам и всем токенам)
            // KV = sum_{r,t} k_phi^T v (для каждого токена)
            let mut kv = vec![0.0f32; d * d];
            let mut z = vec![0.0f32; d];
            for r in 0..batch {
                for t in 0..seq {
                    let idx = r * tokens_per_batch + t * d;
                    // k_phi[idx..idx+d], v[idx..idx+d]
                    for i in 0..d {
                        let ki = k_phi[idx + i];
                        let vi = v_raw[idx + i];
                        z[i] += ki;
                        for j in 0..d {
                            kv[i * d + j] += ki * v_raw[idx + j];
                        }
                    }
                }
            }

            // Вычисляем внимание для каждого токена
            let mut attn_out = vec![0.0f32; batch * tokens_per_batch];
            let eps = 1e-6f32;

            for r in 0..batch {
                for t in 0..seq {
                    let idx = r * tokens_per_batch + t * d;
                    // numerator = sum_i q_phi[i] * kv[i, j]
                    // denominator = sum_i q_phi[i] * z[i] + eps
                    let mut denom = eps;
                    for i in 0..d {
                        denom += q_phi[idx + i] * z[i];
                    }
                    for j in 0..d {
                        let mut num = 0.0;
                        for i in 0..d {
                            num += q_phi[idx + i] * kv[i * d + j];
                        }
                        attn_out[idx + j] = num / denom;
                    }
                }
            }

            // Выходной линейный слой: Y = attn_out W_o + b_o
            // Записываем результат в column-major выход
            for r in 0..batch {
                for t in 0..seq {
                    for j in 0..d {
                        let mut sum = p[bo_start + j];
                        let idx = r * tokens_per_batch + t * d;
                        for i in 0..d {
                            sum += attn_out[idx + i] * p[wo_start + j * d + i];
                        }
                        let out_idx = (t * d + j) * batch + r;
                        y[out_idx] = sum;
                    }
                }
            }

            // Сохраняем кэш для обратного прохода
            self.store_cache(LinearAttentionCache {
                q: q_phi,   // уже после phi
                k: k_phi,   // уже после phi
                v: v_raw,   // без phi
                kv,
                z,
                attn_out,
                batch,
                seq,
                d_model: d,
            });
        });
    }

    fn backward_buffered(
        &self,
        ctx: &DynamicContext,
        grad_output: &MatrixBufferHandle,
        grad_input: &MatrixBufferHandle,
        params: &MatrixBufferHandle,
        slice: &ParamSlice,
        grad_params: &MatrixBufferHandle,
    ) {
        let DynamicContext::Buffered(bc) = ctx;
        let input_handle = match bc {
            BufferedContext::LinearAttention { input } => input,
            _ => panic!("Expected LinearAttention context"),
        };

        let cache = self
            .take_cache()
            .expect("LinearAttention backward called without forward cache");

        let batch = grad_output.rows();
        let seq = self.seq_len;
        let d = self.d_model;
        let tokens_per_batch = seq * d;
        let total_tokens = batch * tokens_per_batch;

        debug_assert_eq!(grad_output.cols(), tokens_per_batch);
        debug_assert_eq!(grad_input.rows(), batch);
        debug_assert_eq!(grad_input.cols(), tokens_per_batch);
        debug_assert_eq!(batch, cache.batch);
        debug_assert_eq!(seq, cache.seq);
        debug_assert_eq!(d, cache.d_model);
        debug_assert!(
            slice.start + self.param_len() <= params.rows() * params.cols(),
            "LinearAttention backward: parameter slice out of bounds"
        );
        debug_assert!(
            slice.start + self.param_len() <= grad_params.rows() * grad_params.cols(),
            "LinearAttention backward: grad parameter slice out of bounds"
        );

        let ids = [
            input_handle.id(),
            grad_output.id(),
            grad_input.id(),
            params.id(),
            grad_params.id(),
        ];

        input_handle
            .memory()
            .write()
            .unwrap()
            .with_cpu_slices_mut(&ids, |slices| {
                let (first, rest) = slices.split_at_mut(1);
                let x: &[f32] = &*first[0];
                let (second, rest) = rest.split_at_mut(1);
                let go: &[f32] = &*second[0];
                let (third, rest) = rest.split_at_mut(1);
                let gi: &mut [f32] = &mut *third[0];
                let (fourth, rest) = rest.split_at_mut(1);
                let p: &[f32] = &*fourth[0];
                let gp: &mut [f32] = &mut *rest[0];

                let base = slice.start;
                let wq_start = base;
                let bq_start = wq_start + d * d;
                let wk_start = bq_start + d;
                let bk_start = wk_start + d * d;
                let wv_start = bk_start + d;
                let bv_start = wv_start + d * d;
                let wo_start = bv_start + d;
                let bo_start = wo_start + d * d;

                // Инициализируем градиенты параметров нулями
                for i in 0..self.param_len() {
                    gp[base + i] = 0.0;
                }

                // Локальные накопители градиентов
                let mut grad_wq = vec![0.0f32; d * d];
                let mut grad_bq = vec![0.0f32; d];
                let mut grad_wk = vec![0.0f32; d * d];
                let mut grad_bk = vec![0.0f32; d];
                let mut grad_wv = vec![0.0f32; d * d];
                let mut grad_bv = vec![0.0f32; d];
                let mut grad_wo = vec![0.0f32; d * d];
                let mut grad_bo = vec![0.0f32; d];

                // Градиент по входу (в column-major) обнуляем
                for i in 0..(batch * tokens_per_batch) {
                    gi[i] = 0.0;
                }

                // Преобразуем вход и градиент выхода в row-major для удобства
                let mut x_rows = vec![0.0f32; batch * tokens_per_batch];
                let mut go_rows = vec![0.0f32; batch * tokens_per_batch];
                for r in 0..batch {
                    for t in 0..seq {
                        for j in 0..d {
                            let idx_src = (t * d + j) * batch + r;
                            let idx_dst = r * tokens_per_batch + t * d + j;
                            x_rows[idx_dst] = x[idx_src];
                            go_rows[idx_dst] = go[idx_src];
                        }
                    }
                }

                // Шаг 1: градиенты по attn_out от выходного линейного слоя.
                // dL/d(attn_out_i) = sum_j go_j * W_o[j,i]
                let mut d_attn_out = vec![0.0f32; batch * tokens_per_batch];
                for idx in 0..(batch * tokens_per_batch) {
                    let mut sum = 0.0;
                    for j in 0..d {
                        sum += go_rows[idx + j] * p[wo_start + j * d + (idx % d)];
                    }
                    // Это неверно для векторной формы; нужно аккуратнее.
                    // Лучше сделать циклы по r,t,i.
                }
                // Перепишем более явно:
                for r in 0..batch {
                    for t in 0..seq {
                        let idx = r * tokens_per_batch + t * d;
                        for i in 0..d {
                            let mut sum = 0.0;
                            for j in 0..d {
                                sum += go_rows[idx + j] * p[wo_start + j * d + i];
                            }
                            d_attn_out[idx + i] = sum;
                        }
                    }
                }

                // Градиенты по W_o и b_o
                for r in 0..batch {
                    for t in 0..seq {
                        let idx = r * tokens_per_batch + t * d;
                        for j in 0..d {
                            let gout = go_rows[idx + j];
                            grad_bo[j] += gout;
                            for i in 0..d {
                                grad_wo[j * d + i] += gout * cache.attn_out[idx + i];
                            }
                        }
                    }
                }

                // Теперь нужно градиенты по q_phi, k_phi, v через вычисление внимания.
                // Это сложная часть. Используем формулы.
                // Для каждого токена (r,t):
                // y_i = (sum_l q_l * kv[l,i]) / denom
                // denom = sum_l q_l * z_l + eps
                // dL/dq_l = sum_i dL/dy_i * (kv[l,i] * denom - y_i * z_l) / denom^2
                // dL/dkv[l,i] = sum_i dL/dy_i * q_l / denom
                // dL/dz_l = sum_i dL/dy_i * (- y_i * q_l) / denom
                // Но kv и z являются суммами по всем токенам, поэтому нужно накапливать.

                let mut d_q_phi = vec![0.0f32; batch * tokens_per_batch];
                let mut d_k_phi = vec![0.0f32; batch * tokens_per_batch];
                let mut d_v = vec![0.0f32; batch * tokens_per_batch];
                let mut d_kv = vec![0.0f32; d * d];
                let mut d_z = vec![0.0f32; d];

                let eps = 1e-6f32;

                // Сначала для каждого токена вычисляем производные по q, kv, z
                for r in 0..batch {
                    for t in 0..seq {
                        let idx = r * tokens_per_batch + t * d;
                        // denom и y уже не сохранены, но мы можем пересчитать
                        let mut denom = eps;
                        for l in 0..d {
                            denom += cache.q[idx + l] * cache.z[l];
                        }
                        let inv_denom = 1.0 / denom;
                        let denom_sq_inv = inv_denom * inv_denom;

                        // Производная по q_l
                        for l in 0..d {
                            let mut grad_q_l = 0.0;
                            for i in 0..d {
                                let dy_i = d_attn_out[idx + i];
                                grad_q_l += dy_i * (cache.kv[l * d + i] * inv_denom
                                    - cache.attn_out[idx + i] * cache.z[l] * denom_sq_inv);
                            }
                            d_q_phi[idx + l] = grad_q_l;
                        }

                        // Производная по kv[l,i] (накапливаем глобально)
                        for l in 0..d {
                            for i in 0..d {
                                d_kv[l * d + i] += d_attn_out[idx + i] * cache.q[idx + l] * inv_denom;
                            }
                        }

                        // Производная по z[l]
                        for l in 0..d {
                            let mut grad_z_l = 0.0;
                            for i in 0..d {
                                grad_z_l += d_attn_out[idx + i] * (-cache.attn_out[idx + i] * cache.q[idx + l] * denom_sq_inv);
                            }
                            d_z[l] += grad_z_l;
                        }
                    }
                }

                // Теперь d_kv и d_z распределяются по k_phi и v.
                // kv = sum_{r,t} k_phi^T * v
                // z = sum k_phi
                // Поэтому:
                // d_k_phi[r,t,l] += sum_i d_kv[l,i] * v[r,t,i] + d_z[l]
                // d_v[r,t,i] += sum_l d_kv[l,i] * k_phi[r,t,l]
                for r in 0..batch {
                    for t in 0..seq {
                        let idx = r * tokens_per_batch + t * d;
                        for l in 0..d {
                            let mut grad_k_l = d_z[l];
                            for i in 0..d {
                                grad_k_l += d_kv[l * d + i] * cache.v[idx + i];
                            }
                            d_k_phi[idx + l] = grad_k_l;
                        }
                        for i in 0..d {
                            let mut grad_v_i = 0.0;
                            for l in 0..d {
                                grad_v_i += d_kv[l * d + i] * cache.k[idx + l];
                            }
                            d_v[idx + i] = grad_v_i;
                        }
                    }
                }

                // Градиенты через phi для q_phi и k_phi
                // d_q_raw = d_q_phi * phi'(q_raw)
                // d_k_raw = d_k_phi * phi'(k_raw)
                // Для этого нужно сохранить q_raw и k_raw, но мы их не сохранили.
                // Можно пересчитать q_raw и k_raw из x и параметров, так как параметры у нас есть.
                // Мы можем вычислить на лету.
                // Но проще: мы не сохранили q_raw, k_raw. Поэтому мы должны их пересчитать.
                // Сделаем это, пройдя ещё раз по данным.
                let mut d_q_raw = vec![0.0f32; batch * tokens_per_batch];
                let mut d_k_raw = vec![0.0f32; batch * tokens_per_batch];
                for r in 0..batch {
                    for t in 0..seq {
                        let idx = r * tokens_per_batch + t * d;
                        for i in 0..d {
                            // пересчитываем q_raw и k_raw
                            let mut q_raw = p[bq_start + i];
                            let mut k_raw = p[bk_start + i];
                            for j in 0..d {
                                q_raw += x_rows[idx + j] * p[wq_start + i * d + j];
                                k_raw += x_rows[idx + j] * p[wk_start + i * d + j];
                            }
                            d_q_raw[idx + i] = d_q_phi[idx + i] * phi_derivative(q_raw);
                            d_k_raw[idx + i] = d_k_phi[idx + i] * phi_derivative(k_raw);
                        }
                    }
                }

                // Градиенты по W_q, b_q, W_k, b_k, W_v, b_v и по входу x
                // d_q_raw = x W_q + b_q => dW_q = x^T d_q_raw, db_q = sum d_q_raw
                // аналогично для k, v.
                // Входной x в row-major.
                // Градиент по входу: dx = d_q_raw W_q^T + d_k_raw W_k^T + d_v_raw W_v^T
                // Но v не проходит через phi, поэтому d_v = d_v напрямую.
                // У нас уже есть d_v.

                for r in 0..batch {
                    for t in 0..seq {
                        let idx = r * tokens_per_batch + t * d;
                        for i in 0..d {
                            // d_q_raw
                            let dq = d_q_raw[idx + i];
                            grad_bq[i] += dq;
                            for j in 0..d {
                                grad_wq[i * d + j] += dq * x_rows[idx + j];
                                // вклад в градиент по входу (пока накапливаем в gi)
                                gi[(t * d + j) * batch + r] += dq * p[wq_start + i * d + j];
                            }

                            // d_k_raw
                            let dk = d_k_raw[idx + i];
                            grad_bk[i] += dk;
                            for j in 0..d {
                                grad_wk[i * d + j] += dk * x_rows[idx + j];
                                gi[(t * d + j) * batch + r] += dk * p[wk_start + i * d + j];
                            }

                            // d_v
                            let dv = d_v[idx + i];
                            grad_bv[i] += dv;
                            for j in 0..d {
                                grad_wv[i * d + j] += dv * x_rows[idx + j];
                                gi[(t * d + j) * batch + r] += dv * p[wv_start + i * d + j];
                            }
                        }
                    }
                }

                // Записываем градиенты параметров
                for i in 0..d {
                    gp[bq_start + i] = grad_bq[i];
                    gp[bk_start + i] = grad_bk[i];
                    gp[bv_start + i] = grad_bv[i];
                    gp[bo_start + i] = grad_bo[i];
                }
                for i in 0..d*d {
                    gp[wq_start + i] = grad_wq[i];
                    gp[wk_start + i] = grad_wk[i];
                    gp[wv_start + i] = grad_wv[i];
                    gp[wo_start + i] = grad_wo[i];
                }
            });
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