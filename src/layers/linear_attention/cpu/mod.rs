// src/layers/linear_attention/cpu/mod.rs

use crate::compute_manager::graph::types::DynamicContext;
use crate::compute_manager::matrix_buffer::MatrixBufferHandle;
use crate::layers::buffered_context::BufferedContext;
use crate::layers::UniversalLayerBuffered;
use crate::model_plan::param_store::ParamSlice;

use super::super::linear_attention::linear_attention::{LinearAttention, LinearAttentionCache};

// ====================== Вспомогательные функции ======================

fn phi(x: f32) -> f32 {
    if x > 0.0 { x + 1.0 } else { x.exp() + 1.0 }
}

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
            let wq_start = base;
            let bq_start = wq_start + d * d;
            let wk_start = bq_start + d;
            let bk_start = wk_start + d * d;
            let wv_start = bk_start + d;
            let bv_start = wv_start + d * d;
            let wo_start = bv_start + d;
            let bo_start = wo_start + d * d;

            let tokens_per_batch = seq * d;
            let mut x_rows = vec![0.0f32; batch * tokens_per_batch];
            for r in 0..batch {
                for t in 0..seq {
                    for j in 0..d {
                        let src_idx = (t * d + j) * batch + r;
                        let dst_idx = r * tokens_per_batch + t * d + j;
                        x_rows[dst_idx] = x[src_idx];
                    }
                }
            }

            let mut q_raw = vec![0.0f32; batch * tokens_per_batch];
            let mut k_raw = vec![0.0f32; batch * tokens_per_batch];
            let mut v_raw = vec![0.0f32; batch * tokens_per_batch];

            for r in 0..batch {
                for t in 0..seq {
                    let offset = r * tokens_per_batch + t * d;
                    for j in 0..d {
                        let mut sum_q = p[bq_start + j];
                        let mut sum_k = p[bk_start + j];
                        let mut sum_v = p[bv_start + j];
                        for i in 0..d {
                            let xv = x_rows[offset + i];
                            sum_q += xv * p[wq_start + j * d + i];
                            sum_k += xv * p[wk_start + j * d + i];
                            sum_v += xv * p[wv_start + j * d + i];
                        }
                        q_raw[offset + j] = sum_q;
                        k_raw[offset + j] = sum_k;
                        v_raw[offset + j] = sum_v;
                    }
                }
            }

            let mut q_phi = vec![0.0f32; batch * tokens_per_batch];
            let mut k_phi = vec![0.0f32; batch * tokens_per_batch];
            for i in 0..q_raw.len() {
                q_phi[i] = phi(q_raw[i]);
                k_phi[i] = phi(k_raw[i]);
            }

            let mut kv = vec![0.0f32; d * d];
            let mut z = vec![0.0f32; d];
            for r in 0..batch {
                for t in 0..seq {
                    let idx = r * tokens_per_batch + t * d;
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

            let mut attn_out = vec![0.0f32; batch * tokens_per_batch];
            let eps = 1e-6f32;

            for r in 0..batch {
                for t in 0..seq {
                    let idx = r * tokens_per_batch + t * d;
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

            self.store_cache(LinearAttentionCache {
                q: q_phi,
                k: k_phi,
                v: v_raw,
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

                for i in 0..self.param_len() {
                    gp[base + i] = 0.0;
                }

                let mut grad_wq = vec![0.0f32; d * d];
                let mut grad_bq = vec![0.0f32; d];
                let mut grad_wk = vec![0.0f32; d * d];
                let mut grad_bk = vec![0.0f32; d];
                let mut grad_wv = vec![0.0f32; d * d];
                let mut grad_bv = vec![0.0f32; d];
                let mut grad_wo = vec![0.0f32; d * d];
                let mut grad_bo = vec![0.0f32; d];

                for i in 0..(batch * tokens_per_batch) {
                    gi[i] = 0.0;
                }

                let mut x_rows = vec![0.0f32; batch * tokens_per_batch];
                let mut go_rows = vec![0.0f32; batch * tokens_per_batch];
                for r in 0..batch {
                    for t in 0..seq {
                        for j in 0..d {
                            let src_idx = (t * d + j) * batch + r;
                            let dst_idx = r * tokens_per_batch + t * d + j;
                            x_rows[dst_idx] = x[src_idx];
                            go_rows[dst_idx] = go[src_idx];
                        }
                    }
                }

                let mut d_attn_out = vec![0.0f32; batch * tokens_per_batch];
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

                let mut d_q_phi = vec![0.0f32; batch * tokens_per_batch];
                let mut d_k_phi = vec![0.0f32; batch * tokens_per_batch];
                let mut d_v = vec![0.0f32; batch * tokens_per_batch];
                let mut d_kv = vec![0.0f32; d * d];
                let mut d_z = vec![0.0f32; d];

                let eps = 1e-6f32;

                for r in 0..batch {
                    for t in 0..seq {
                        let idx = r * tokens_per_batch + t * d;
                        let mut denom = eps;
                        for l in 0..d {
                            denom += cache.q[idx + l] * cache.z[l];
                        }
                        let inv_denom = 1.0 / denom;
                        let denom_sq_inv = inv_denom * inv_denom;

                        for l in 0..d {
                            let mut grad_q_l = 0.0;
                            for i in 0..d {
                                let dy_i = d_attn_out[idx + i];
                                grad_q_l += dy_i * (cache.kv[l * d + i] * inv_denom
                                    - cache.attn_out[idx + i] * cache.z[l] * denom_sq_inv);
                            }
                            d_q_phi[idx + l] = grad_q_l;
                        }

                        for l in 0..d {
                            for i in 0..d {
                                d_kv[l * d + i] += d_attn_out[idx + i] * cache.q[idx + l] * inv_denom;
                            }
                        }

                        for l in 0..d {
                            let mut grad_z_l = 0.0;
                            for i in 0..d {
                                grad_z_l += d_attn_out[idx + i] * (-cache.attn_out[idx + i] * cache.q[idx + l] * denom_sq_inv);
                            }
                            d_z[l] += grad_z_l;
                        }
                    }
                }

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

                let mut d_q_raw = vec![0.0f32; batch * tokens_per_batch];
                let mut d_k_raw = vec![0.0f32; batch * tokens_per_batch];
                for r in 0..batch {
                    for t in 0..seq {
                        let idx = r * tokens_per_batch + t * d;
                        for i in 0..d {
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

                for r in 0..batch {
                    for t in 0..seq {
                        let idx = r * tokens_per_batch + t * d;
                        for i in 0..d {
                            let dq = d_q_raw[idx + i];
                            grad_bq[i] += dq;
                            for j in 0..d {
                                grad_wq[i * d + j] += dq * x_rows[idx + j];
                                gi[(t * d + j) * batch + r] += dq * p[wq_start + i * d + j];
                            }

                            let dk = d_k_raw[idx + i];
                            grad_bk[i] += dk;
                            for j in 0..d {
                                grad_wk[i * d + j] += dk * x_rows[idx + j];
                                gi[(t * d + j) * batch + r] += dk * p[wk_start + i * d + j];
                            }

                            let dv = d_v[idx + i];
                            grad_bv[i] += dv;
                            for j in 0..d {
                                grad_wv[i * d + j] += dv * x_rows[idx + j];
                                gi[(t * d + j) * batch + r] += dv * p[wv_start + i * d + j];
                            }
                        }
                    }
                }

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