// src/layers/relative_position_attention/cpu/mod.rs

use crate::compute_manager::graph::types::DynamicContext;
use crate::compute_manager::matrix_buffer::MatrixBufferHandle;
use crate::layers::buffered_context::BufferedContext;
use crate::layers::UniversalLayerBuffered;
use crate::model_plan::param_store::ParamSlice;

use super::super::relative_position_attention::relative_position_attention::{
    RelativePositionAttention, RelativePositionAttentionCache,
};

/// Создаёт пустой кэш RelativePositionAttentionCache для std::mem::replace.
fn empty_relative_position_attention_cache() -> RelativePositionAttentionCache {
    RelativePositionAttentionCache {
        q: Vec::new(),
        k: Vec::new(),
        v: Vec::new(),
        scores: Vec::new(),
        attention_weights: Vec::new(),
        attn_out: Vec::new(),
        batch: 0,
        seq: 0,
        d_model: 0,
    }
}

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
        let features = seq * d;
        let total = batch * features;

        debug_assert_eq!(input.cols(), features);
        debug_assert_eq!(output.rows(), batch);
        debug_assert_eq!(output.cols(), features);
        debug_assert!(slice.start + self.param_len() <= params.rows() * params.cols());

        let (q_vec, k_vec, v_vec, scores_vec, weights_vec, attn_out_vec) = {
            let ids = [input.id(), output.id(), params.id()];
            input
                .memory()
                .write()
                .unwrap()
                .with_cpu_slices_mut(&ids, |slices| {
                    let (first, rest) = slices.split_at_mut(1);
                    let x: &[f32] = &*first[0];
                    let (second, rest) = rest.split_at_mut(1);
                    let y: &mut [f32] = &mut *second[0];
                    let p: &[f32] = &*rest[0];

                    let base = slice.start;
                    let wq_start = base;
                    let bq_start = wq_start + d * d;
                    let wk_start = bq_start + d;
                    let bk_start = wk_start + d * d;
                    let wv_start = bk_start + d;
                    let bv_start = wv_start + d * d;
                    let wo_start = bv_start + d;
                    let bo_start = wo_start + d * d;
                    let rel_bias_start = bo_start + d;

                    // ============ 1. QKV-проекции (column-major) ============
                    let mut q = vec![0.0f32; total];
                    let mut k = vec![0.0f32; total];
                    let mut v = vec![0.0f32; total];

                    for r in 0..batch {
                        for t in 0..seq {
                            let tok_base = (t * d) * batch + r;
                            for j in 0..d {
                                let mut sq = p[bq_start + j];
                                let mut sk = p[bk_start + j];
                                let mut sv = p[bv_start + j];
                                for i in 0..d {
                                    let xv = x[tok_base + i * batch];
                                    sq += xv * p[wq_start + j * d + i];
                                    sk += xv * p[wk_start + j * d + i];
                                    sv += xv * p[wv_start + j * d + i];
                                }
                                let idx = tok_base + j * batch;
                                q[idx] = sq;
                                k[idx] = sk;
                                v[idx] = sv;
                            }
                        }
                    }

                    // ============ 2. Scores и softmax ============
                    let scale = 1.0f32 / (d as f32).sqrt();
                    let scores_total = batch * seq * seq;

                    let mut scores = vec![0.0f32; scores_total];
                    let mut weights = vec![0.0f32; scores_total];

                    for r in 0..batch {
                        for t in 0..seq {
                            let q_base = (t * d) * batch + r;
                            let s_base = (t * seq) * batch + r;

                            let mut max_score = f32::NEG_INFINITY;
                            for s in 0..seq {
                                let k_base = (s * d) * batch + r;
                                let mut dot = 0.0f32;
                                for j in 0..d {
                                    dot += q[q_base + j * batch]
                                        * k[k_base + j * batch];
                                }
                                dot *= scale;

                                let rel_idx =
                                    (s as isize - t as isize + (seq as isize - 1))
                                        as usize;
                                let score = dot + p[rel_bias_start + rel_idx];
                                scores[s_base + s * batch] = score;
                                if score > max_score {
                                    max_score = score;
                                }
                            }

                            let mut sum_exp = 0.0f32;
                            for s in 0..seq {
                                sum_exp += (scores[s_base + s * batch] - max_score).exp();
                            }
                            let inv_sum = 1.0 / sum_exp;
                            for s in 0..seq {
                                let e =
                                    (scores[s_base + s * batch] - max_score).exp();
                                weights[s_base + s * batch] = e * inv_sum;
                            }
                        }
                    }

                    // ============ 3. attn_out ============
                    let mut attn_out = vec![0.0f32; total];

                    for r in 0..batch {
                        for t in 0..seq {
                            let w_base = (t * seq) * batch + r;
                            let a_base = (t * d) * batch + r;
                            for i in 0..d {
                                let mut sum = 0.0f32;
                                for s in 0..seq {
                                    let v_idx = (s * d + i) * batch + r;
                                    sum += weights[w_base + s * batch] * v[v_idx];
                                }
                                attn_out[a_base + i * batch] = sum;
                            }
                        }
                    }

                    // ============ 4. Выходной линейный слой ============
                    for r in 0..batch {
                        for t in 0..seq {
                            let a_base = (t * d) * batch + r;
                            for j in 0..d {
                                let mut sum = p[bo_start + j];
                                for i in 0..d {
                                    sum += attn_out[a_base + i * batch]
                                        * p[wo_start + j * d + i];
                                }
                                y[a_base + j * batch] = sum;
                            }
                        }
                    }

                    (q, k, v, scores, weights, attn_out)
                })
        };

        self.store_cache(RelativePositionAttentionCache {
            q: q_vec,
            k: k_vec,
            v: v_vec,
            scores: scores_vec,
            attention_weights: weights_vec,
            attn_out: attn_out_vec,
            batch,
            seq,
            d_model: d,
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
            BufferedContext::RelativePositionAttention { input, .. } => input,
            _ => panic!("Expected RelativePositionAttention context"),
        };

        let batch = grad_output.rows();
        let seq = self.seq_len;
        let d = self.d_model;
        let features = seq * d;
        let total = batch * features;
        let scores_total = batch * seq * seq;

        debug_assert_eq!(grad_output.cols(), features);
        debug_assert_eq!(grad_input.rows(), batch);
        debug_assert_eq!(grad_input.cols(), features);
        debug_assert!(
            slice.start + self.param_len() <= params.rows() * params.cols(),
            "RelativePositionAttention backward: parameter slice out of bounds"
        );
        debug_assert!(
            slice.start + self.param_len() <= grad_params.rows() * grad_params.cols(),
            "RelativePositionAttention backward: grad parameter slice out of bounds"
        );

        // Извлекаем кэш и инвалидируем состояние.
        // std::mem::replace позволяет избежать клонирования больших векторов.
        let cache = {
            let mut guard = self.state.write().unwrap();
            assert!(
                guard.valid,
                "RelativePositionAttention backward called without forward cache"
            );
            guard.valid = false;
            std::mem::replace(
                &mut guard.cache,
                empty_relative_position_attention_cache(),
            )
        };

        debug_assert_eq!(batch, cache.batch);
        debug_assert_eq!(seq, cache.seq);
        debug_assert_eq!(d, cache.d_model);

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
                let rel_bias_start = bo_start + d;

                // Обнуление градиентов.
                for i in 0..self.param_len() {
                    gp[base + i] = 0.0;
                }
                for i in 0..total {
                    gi[i] = 0.0;
                }

                let mut grad_wq = vec![0.0f32; d * d];
                let mut grad_bq = vec![0.0f32; d];
                let mut grad_wk = vec![0.0f32; d * d];
                let mut grad_bk = vec![0.0f32; d];
                let mut grad_wv = vec![0.0f32; d * d];
                let mut grad_bv = vec![0.0f32; d];
                let mut grad_wo = vec![0.0f32; d * d];
                let mut grad_bo = vec![0.0f32; d];
                let mut grad_rel_bias = vec![0.0f32; 2 * seq - 1];

                // ============ 1. grad_Wo, grad_bo, d_attn_out ============
                let mut d_attn_out = vec![0.0f32; total];

                for r in 0..batch {
                    for t in 0..seq {
                        let tok_base = (t * d) * batch + r;

                        for j in 0..d {
                            let go_j = go[tok_base + j * batch];
                            grad_bo[j] += go_j;
                            for i in 0..d {
                                grad_wo[j * d + i] +=
                                    go_j * cache.attn_out[tok_base + i * batch];
                            }
                        }

                        for i in 0..d {
                            let mut sum_j = 0.0f32;
                            for j in 0..d {
                                sum_j += go[tok_base + j * batch]
                                    * p[wo_start + j * d + i];
                            }
                            d_attn_out[tok_base + i * batch] = sum_j;
                        }
                    }
                }

                // ============ 2. d_weights, d_v ============
                let mut d_weights = vec![0.0f32; scores_total];
                let mut d_v = vec![0.0f32; total];

                for r in 0..batch {
                    for t in 0..seq {
                        let w_base = (t * seq) * batch + r;
                        let a_base = (t * d) * batch + r;
                        for s in 0..seq {
                            let mut sum = 0.0f32;
                            for i in 0..d {
                                let v_idx = (s * d + i) * batch + r;
                                sum += d_attn_out[a_base + i * batch]
                                    * cache.v[v_idx];
                            }
                            d_weights[w_base + s * batch] = sum;
                        }
                    }
                }

                for r in 0..batch {
                    for s in 0..seq {
                        let v_base = (s * d) * batch + r;
                        for i in 0..d {
                            let mut sum = 0.0f32;
                            for t in 0..seq {
                                let w_idx = (t * seq + s) * batch + r;
                                let a_idx = (t * d + i) * batch + r;
                                sum += d_attn_out[a_idx]
                                    * cache.attention_weights[w_idx];
                            }
                            d_v[v_base + i * batch] = sum;
                        }
                    }
                }

                // ============ 3. d_scores (softmax backward) ============
                let mut d_scores = vec![0.0f32; scores_total];

                for r in 0..batch {
                    for t in 0..seq {
                        let s_base = (t * seq) * batch + r;
                        let mut dot = 0.0f32;
                        for s in 0..seq {
                            dot += cache.attention_weights[s_base + s * batch]
                                * d_weights[s_base + s * batch];
                        }
                        for s in 0..seq {
                            let w = cache.attention_weights[s_base + s * batch];
                            let dw = d_weights[s_base + s * batch];
                            d_scores[s_base + s * batch] = w * (dw - dot);
                        }
                    }
                }

                // ============ 4. d_q, d_k, grad_rel_bias ============
                let scale = 1.0f32 / (d as f32).sqrt();
                let mut d_q = vec![0.0f32; total];
                let mut d_k = vec![0.0f32; total];

                for r in 0..batch {
                    for t in 0..seq {
                        let q_base = (t * d) * batch + r;
                        let ds_base = (t * seq) * batch + r;
                        for s in 0..seq {
                            let ds = d_scores[ds_base + s * batch];
                            let k_base = (s * d) * batch + r;

                            for i in 0..d {
                                d_q[q_base + i * batch] +=
                                    ds * cache.k[k_base + i * batch] * scale;
                                d_k[k_base + i * batch] +=
                                    ds * cache.q[q_base + i * batch] * scale;
                            }

                            let rel_idx =
                                (s as isize - t as isize + (seq as isize - 1)) as usize;
                            grad_rel_bias[rel_idx] += ds;
                        }
                    }
                }

                // ============ 5. gi и градиенты параметров Q, K, V ============
                for r in 0..batch {
                    for t in 0..seq {
                        let tok_base = (t * d) * batch + r;
                        for j in 0..d {
                            let dq = d_q[tok_base + j * batch];
                            let dk = d_k[tok_base + j * batch];
                            let dv = d_v[tok_base + j * batch];

                            grad_bq[j] += dq;
                            grad_bk[j] += dk;
                            grad_bv[j] += dv;

                            for i in 0..d {
                                let xv = x[tok_base + i * batch];

                                grad_wq[j * d + i] += dq * xv;
                                grad_wk[j * d + i] += dk * xv;
                                grad_wv[j * d + i] += dv * xv;

                                gi[tok_base + i * batch] +=
                                    dq * p[wq_start + j * d + i]
                                    + dk * p[wk_start + j * d + i]
                                    + dv * p[wv_start + j * d + i];
                            }
                        }
                    }
                }

                // ============ 6. Запись градиентов параметров ============
                for i in 0..d {
                    gp[bq_start + i] = grad_bq[i];
                    gp[bk_start + i] = grad_bk[i];
                    gp[bv_start + i] = grad_bv[i];
                    gp[bo_start + i] = grad_bo[i];
                }
                for i in 0..d * d {
                    gp[wq_start + i] = grad_wq[i];
                    gp[wk_start + i] = grad_wk[i];
                    gp[wv_start + i] = grad_wv[i];
                    gp[wo_start + i] = grad_wo[i];
                }
                for i in 0..(2 * seq - 1) {
                    gp[rel_bias_start + i] = grad_rel_bias[i];
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