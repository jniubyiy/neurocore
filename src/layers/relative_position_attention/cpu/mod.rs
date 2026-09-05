// src/layers/relative_position_attention/cpu/mod.rs

use crate::compute_manager::graph::types::DynamicContext;
use crate::compute_manager::matrix_buffer::MatrixBufferHandle;
use crate::layers::buffered_context::BufferedContext;
use crate::layers::UniversalLayerBuffered;
use crate::model_plan::param_store::ParamSlice;

use super::super::relative_position_attention::{RelativePositionAttention, RelativePositionAttentionCache};

impl UniversalLayerBuffered for RelativePositionAttention {
    fn forward_buffered(
        &self,
        input: &MatrixBufferHandle,
        output: &MatrixBufferHandle,
        params: &MatrixBufferHandle,
        slice: &ParamSlice,
    ) {
        let batch = input.rows();
        let seq = self.seq_len;
        let d = self.d_model;
        let total_tokens = seq * d;

        debug_assert_eq!(input.cols(), total_tokens);
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
            let bias_start = bo_start + d; // relative_bias (2*seq-1)

            // Преобразуем вход в row-major: (batch, seq*d)
            let mut x_rows = vec![0.0f32; batch * total_tokens];
            for r in 0..batch {
                for t in 0..seq {
                    for j in 0..d {
                        let src_idx = (t * d + j) * batch + r;
                        let dst_idx = r * total_tokens + t * d + j;
                        x_rows[dst_idx] = x[src_idx];
                    }
                }
            }

            // Вычисляем Q, K, V
            let mut q = vec![0.0f32; batch * total_tokens];
            let mut k = vec![0.0f32; batch * total_tokens];
            let mut v = vec![0.0f32; batch * total_tokens];
            for r in 0..batch {
                for t in 0..seq {
                    let offset = r * total_tokens + t * d;
                    for j in 0..d {
                        let mut sum_q = p[bq_start + j];
                        let mut sum_k = p[bk_start + j];
                        let mut sum_v = p[bv_start + j];
                        for i in 0..d {
                            let x_val = x_rows[offset + i];
                            sum_q += x_val * p[wq_start + j * d + i];
                            sum_k += x_val * p[wk_start + j * d + i];
                            sum_v += x_val * p[wv_start + j * d + i];
                        }
                        q[offset + j] = sum_q;
                        k[offset + j] = sum_k;
                        v[offset + j] = sum_v;
                    }
                }
            }

            // Вычисляем скоры и softmax
            let scale = 1.0f32 / (d as f32).sqrt();
            let mut scores = vec![0.0f32; batch * seq * seq];
            let mut attention_weights = vec![0.0f32; batch * seq * seq];

            for r in 0..batch {
                for t in 0..seq {
                    let q_offset = r * total_tokens + t * d;
                    let score_offset = r * seq * seq + t * seq;
                    // Сначала вычисляем все скоры
                    let mut max_score = f32::NEG_INFINITY;
                    for s in 0..seq {
                        let k_offset = r * total_tokens + s * d;
                        let mut score = 0.0;
                        for j in 0..d {
                            score += q[q_offset + j] * k[k_offset + j];
                        }
                        score *= scale;
                        // добавляем относительное смещение
                        let rel_idx = (s as isize - t as isize + (seq as isize - 1)) as usize;
                        score += p[bias_start + rel_idx];
                        scores[score_offset + s] = score;
                        if score > max_score { max_score = score; }
                    }
                    // softmax
                    let mut sum_exp = 0.0;
                    let mut exps = vec![0.0f32; seq];
                    for s in 0..seq {
                        let e = (scores[score_offset + s] - max_score).exp();
                        exps[s] = e;
                        sum_exp += e;
                    }
                    for s in 0..seq {
                        attention_weights[score_offset + s] = exps[s] / sum_exp;
                    }
                }
            }

            // Вычисляем attn_out
            let mut attn_out = vec![0.0f32; batch * total_tokens];
            for r in 0..batch {
                for t in 0..seq {
                    let out_offset = r * total_tokens + t * d;
                    let weight_offset = r * seq * seq + t * seq;
                    for j in 0..d {
                        let mut sum = 0.0;
                        for s in 0..seq {
                            let v_offset = r * total_tokens + s * d + j;
                            sum += attention_weights[weight_offset + s] * v[v_offset];
                        }
                        attn_out[out_offset + j] = sum;
                    }
                }
            }

            // Выходной линейный слой и запись в column-major
            for r in 0..batch {
                for t in 0..seq {
                    for j in 0..d {
                        let mut sum = p[bo_start + j];
                        let offset = r * total_tokens + t * d;
                        for i in 0..d {
                            sum += attn_out[offset + i] * p[wo_start + j * d + i];
                        }
                        let out_idx = (t * d + j) * batch + r;
                        y[out_idx] = sum;
                    }
                }
            }

            // Сохраняем кэш
            self.store_cache(RelativePositionAttentionCache {
                q,
                k,
                v,
                scores,
                attention_weights,
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
            BufferedContext::RelativePositionAttention { input } => input,
            _ => panic!("Expected RelativePositionAttention context"),
        };

        let cache = self
            .take_cache()
            .expect("RelativePositionAttention backward called without forward cache");

        let batch = grad_output.rows();
        let seq = self.seq_len;
        let d = self.d_model;
        let total_tokens = seq * d;

        debug_assert_eq!(grad_output.cols(), total_tokens);
        debug_assert_eq!(grad_input.rows(), batch);
        debug_assert_eq!(grad_input.cols(), total_tokens);
        debug_assert_eq!(batch, cache.batch);
        debug_assert_eq!(seq, cache.seq);
        debug_assert_eq!(d, cache.d_model);
        debug_assert!(
            slice.start + self.param_len() <= params.rows() * params.cols(),
            "RelativePositionAttention backward: parameter slice out of bounds"
        );
        debug_assert!(
            slice.start + self.param_len() <= grad_params.rows() * grad_params.cols(),
            "RelativePositionAttention backward: grad parameter slice out of bounds"
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
                let bias_start = bo_start + d;

                // Инициализируем градиенты параметров нулями
                for i in 0..self.param_len() {
                    gp[base + i] = 0.0;
                }

                // Локальные накопители
                let mut grad_wq = vec![0.0f32; d * d];
                let mut grad_bq = vec![0.0f32; d];
                let mut grad_wk = vec![0.0f32; d * d];
                let mut grad_bk = vec![0.0f32; d];
                let mut grad_wv = vec![0.0f32; d * d];
                let mut grad_bv = vec![0.0f32; d];
                let mut grad_wo = vec![0.0f32; d * d];
                let mut grad_bo = vec![0.0f32; d];
                let mut grad_rel_bias = vec![0.0f32; 2 * seq - 1];

                // Градиент по входу (column-major) обнуляем
                for i in 0..(batch * total_tokens) {
                    gi[i] = 0.0;
                }

                // Преобразуем вход и градиент выхода в row-major
                let mut x_rows = vec![0.0f32; batch * total_tokens];
                let mut go_rows = vec![0.0f32; batch * total_tokens];
                for r in 0..batch {
                    for t in 0..seq {
                        for j in 0..d {
                            let src_idx = (t * d + j) * batch + r;
                            let dst_idx = r * total_tokens + t * d + j;
                            x_rows[dst_idx] = x[src_idx];
                            go_rows[dst_idx] = go[src_idx];
                        }
                    }
                }

                // 1. Градиенты по attn_out и параметрам выходного линейного слоя
                let mut d_attn_out = vec![0.0f32; batch * total_tokens];
                for r in 0..batch {
                    for t in 0..seq {
                        let idx = r * total_tokens + t * d;
                        for i in 0..d {
                            let mut grad_i = 0.0;
                            for j in 0..d {
                                let go_val = go_rows[idx + j];
                                grad_i += go_val * p[wo_start + j * d + i];
                                grad_wo[j * d + i] += go_val * cache.attn_out[idx + i];
                                grad_bo[j] += go_val;
                            }
                            d_attn_out[idx + i] = grad_i;
                        }
                    }
                }

                // 2. Градиенты по v и attention_weights
                let mut d_v = vec![0.0f32; batch * total_tokens];
                let mut d_weights = vec![0.0f32; batch * seq * seq];
                for r in 0..batch {
                    for s in 0..seq {
                        for i in 0..d {
                            let v_idx = r * total_tokens + s * d + i;
                            let mut grad_v = 0.0;
                            for t in 0..seq {
                                let w = cache.attention_weights[r * seq * seq + t * seq + s];
                                grad_v += w * d_attn_out[r * total_tokens + t * d + i];
                                d_weights[r * seq * seq + t * seq + s] += d_attn_out[r * total_tokens + t * d + i] * cache.v[v_idx];
                            }
                            d_v[v_idx] = grad_v;
                        }
                    }
                }

                // 3. Градиенты по scores (softmax производная)
                let mut d_scores = vec![0.0f32; batch * seq * seq];
                for r in 0..batch {
                    for t in 0..seq {
                        let weight_offset = r * seq * seq + t * seq;
                        // dot = sum_s weight * d_weight
                        let mut dot = 0.0;
                        for s in 0..seq {
                            dot += cache.attention_weights[weight_offset + s] * d_weights[weight_offset + s];
                        }
                        for s in 0..seq {
                            let w = cache.attention_weights[weight_offset + s];
                            let dw = d_weights[weight_offset + s];
                            d_scores[weight_offset + s] = w * (dw - dot);
                        }
                    }
                }

                // 4. Градиенты по q, k, relative_bias
                let mut d_q = vec![0.0f32; batch * total_tokens];
                let mut d_k = vec![0.0f32; batch * total_tokens];
                let scale = 1.0f32 / (d as f32).sqrt();

                for r in 0..batch {
                    for t in 0..seq {
                        let q_offset = r * total_tokens + t * d;
                        let score_offset = r * seq * seq + t * seq;
                        for s in 0..seq {
                            let ds = d_scores[score_offset + s];
                            let k_offset = r * total_tokens + s * d;
                            for j in 0..d {
                                d_q[q_offset + j] += ds * cache.k[k_offset + j] * scale;
                                d_k[k_offset + j] += ds * cache.q[q_offset + j] * scale;
                            }
                            // relative_bias
                            let rel_idx = (s as isize - t as isize + (seq as isize - 1)) as usize;
                            grad_rel_bias[rel_idx] += ds;
                        }
                    }
                }

                // 5. Градиенты по линейным преобразованиям Q, K, V и по входу
                let mut d_q_raw = d_q; // после phi нет, поэтому совпадают
                let mut d_k_raw = d_k;
                let mut d_v_raw = d_v;

                for r in 0..batch {
                    for t in 0..seq {
                        let idx = r * total_tokens + t * d;
                        for i in 0..d {
                            let dq = d_q_raw[idx + i];
                            grad_bq[i] += dq;
                            for j in 0..d {
                                grad_wq[i * d + j] += dq * x_rows[idx + j];
                                // градиент по входу (column-major)
                                gi[(t * d + j) * batch + r] += dq * p[wq_start + i * d + j];
                            }

                            let dk = d_k_raw[idx + i];
                            grad_bk[i] += dk;
                            for j in 0..d {
                                grad_wk[i * d + j] += dk * x_rows[idx + j];
                                gi[(t * d + j) * batch + r] += dk * p[wk_start + i * d + j];
                            }

                            let dv = d_v_raw[idx + i];
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
                for i in 0..(2*seq-1) {
                    gp[bias_start + i] = grad_rel_bias[i];
                }
            });
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