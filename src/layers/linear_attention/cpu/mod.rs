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

        let (q_phi_vec, k_phi_vec, v_raw_vec, kv_vec, z_vec, attn_out_vec) = {
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

                let total = batch * total_tokens;

                // ================== 1. QKV-проекции ==================
                // Column-major:
                //   q_raw[(t*d + j) * batch + r] = b_q[j] + Σ_i x[(t*d+i)*batch+r] * W_q[j*d+i]
                let mut q_raw = vec![0.0f32; total];
                let mut k_raw = vec![0.0f32; total];
                let mut v_raw = vec![0.0f32; total];

                for r in 0..batch {
                    for t in 0..seq {
                        for j in 0..d {
                            let mut sq = p[bq_start + j];
                            let mut sk = p[bk_start + j];
                            let mut sv = p[bv_start + j];
                            for i in 0..d {
                                let xv = x[(t * d + i) * batch + r];
                                sq += xv * p[wq_start + j * d + i];
                                sk += xv * p[wk_start + j * d + i];
                                sv += xv * p[wv_start + j * d + i];
                            }
                            let idx = (t * d + j) * batch + r;
                            q_raw[idx] = sq;
                            k_raw[idx] = sk;
                            v_raw[idx] = sv;
                        }
                    }
                }

                // ================== 2. φ ==================
                let mut q_phi = vec![0.0f32; total];
                let mut k_phi = vec![0.0f32; total];
                for i in 0..total {
                    q_phi[i] = phi(q_raw[i]);
                    k_phi[i] = phi(k_raw[i]);
                }

                // ================== 3. KV и Z ==================
                // kv[j*d + i] = Σ_{r,t} k_phi[(t*d+i)*batch+r] * v_raw[(t*d+j)*batch+r]
                // z[i]        = Σ_{r,t} k_phi[(t*d+i)*batch+r]
                let mut kv = vec![0.0f32; d * d];
                let mut z = vec![0.0f32; d];

                for r in 0..batch {
                    for t in 0..seq {
                        let tok_base = (t * d) * batch + r;
                        for i in 0..d {
                            let ki = k_phi[tok_base + i * batch];
                            z[i] += ki;
                            for j in 0..d {
                                let vj = v_raw[tok_base + j * batch];
                                kv[j * d + i] += ki * vj;
                            }
                        }
                    }
                }

                // ================== 4. attn_out ==================
                // attn_out[r,t,i] = (Σ_l q_phi[r,t,l] * kv[i*d + l]) / (eps + Σ_l q_phi[r,t,l]*z[l])
                let eps = 1e-6f32;
                let mut attn_out = vec![0.0f32; total];

                for r in 0..batch {
                    for t in 0..seq {
                        let tok_base = (t * d) * batch + r;
                        let mut denom = eps;
                        for l in 0..d {
                            denom += q_phi[tok_base + l * batch] * z[l];
                        }
                        let inv_denom = 1.0 / denom;
                        for i in 0..d {
                            let mut num = 0.0;
                            for l in 0..d {
                                num += q_phi[tok_base + l * batch] * kv[i * d + l];
                            }
                            attn_out[tok_base + i * batch] = num * inv_denom;
                        }
                    }
                }

                // ================== 5. Y ==================
                // y[r,t,j] = b_o[j] + Σ_i attn_out[r,t,i] * W_o[j*d+i]
                for r in 0..batch {
                    for t in 0..seq {
                        let tok_base = (t * d) * batch + r;
                        for j in 0..d {
                            let mut sum = p[bo_start + j];
                            for i in 0..d {
                                sum += attn_out[tok_base + i * batch] * p[wo_start + j * d + i];
                            }
                            y[tok_base + j * batch] = sum;
                        }
                    }
                }

                (q_phi, k_phi, v_raw, kv, z, attn_out)
            })
        };

        self.store_cache(LinearAttentionCache {
            q: q_phi_vec,
            k: k_phi_vec,
            v: v_raw_vec,
            kv: kv_vec,
            z: z_vec,
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
            BufferedContext::LinearAttention { input, .. } => input,
            _ => panic!("Expected LinearAttention context"),
        };

        let cache = self
            .take_cache()
            .expect("LinearAttention backward called without forward cache");

        let batch = grad_output.rows();
        let seq = self.seq_len;
        let d = self.d_model;
        let total_tokens = seq * d;
        let total = batch * total_tokens;

        debug_assert_eq!(grad_output.cols(), total_tokens);
        debug_assert_eq!(grad_input.rows(), batch);
        debug_assert_eq!(grad_input.cols(), total_tokens);
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

                // Обнуляем градиенты параметров и входной градиент.
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

                // ================== 1. d_attn_out, grad_Wo, grad_bo ==================
                // grad_bo[j] = Σ_{r,t} go[r,t,j]
                // grad_Wo[j,i] = Σ_{r,t} go[r,t,j] * attn_out[r,t,i]
                // d_attn_out[r,t,i] = Σ_j go[r,t,j] * W_o[j,i]
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
                            let mut sum_j = 0.0;
                            for j in 0..d {
                                sum_j +=
                                    go[tok_base + j * batch] * p[wo_start + j * d + i];
                            }
                            d_attn_out[tok_base + i * batch] = sum_j;
                        }
                    }
                }

                // ================== 2. d_q_phi, d_kv_local, d_z_local ==================
                // d_q_phi[r,t,l] = Σ_i d_attn_out[r,t,i] *
                //                  ( kv[i*d+l]/denom - attn_out[r,t,i]*z[l]/denom² )
                // d_kv_local[i*d+l] += d_attn_out[r,t,i] * q_phi[r,t,l] / denom
                // d_z_local[l]      += -q_phi[r,t,l]/denom * Σ_i d_attn_out[r,t,i]*attn_out[r,t,i]
                let eps = 1e-6f32;
                let mut d_q_phi = vec![0.0f32; total];
                let mut d_kv_local = vec![0.0f32; d * d];
                let mut d_z_local = vec![0.0f32; d];

                for r in 0..batch {
                    for t in 0..seq {
                        let tok_base = (t * d) * batch + r;

                        let mut denom = eps;
                        for l in 0..d {
                            denom += cache.q[tok_base + l * batch] * cache.z[l];
                        }
                        let inv_denom = 1.0 / denom;
                        let inv_denom_sq = inv_denom * inv_denom;

                        for l in 0..d {
                            let q_l = cache.q[tok_base + l * batch];

                            let mut dq_l = 0.0;
                            let mut sum_da_ao = 0.0;

                            for i in 0..d {
                                let da = d_attn_out[tok_base + i * batch];
                                let ao = cache.attn_out[tok_base + i * batch];
                                let kv_li = cache.kv[i * d + l];

                                dq_l += da
                                    * (kv_li * inv_denom - ao * cache.z[l] * inv_denom_sq);
                                sum_da_ao += da * ao;

                                d_kv_local[i * d + l] += da * q_l * inv_denom;
                            }

                            d_q_phi[tok_base + l * batch] = dq_l;
                            d_z_local[l] += -q_l * inv_denom * sum_da_ao;
                        }
                    }
                }

                // ================== 3. d_k_phi, d_v ==================
                // d_k_phi[r,t,l] = d_z_local[l] + Σ_i d_kv_local[i*d+l] * v[r,t,i]
                // d_v[r,t,i]     = Σ_l d_kv_local[i*d+l] * k_phi[r,t,l]
                let mut d_k_phi = vec![0.0f32; total];
                let mut d_v = vec![0.0f32; total];

                for r in 0..batch {
                    for t in 0..seq {
                        let tok_base = (t * d) * batch + r;

                        for l in 0..d {
                            let mut sum_k = d_z_local[l];
                            for i in 0..d {
                                sum_k += d_kv_local[i * d + l]
                                    * cache.v[tok_base + i * batch];
                            }
                            d_k_phi[tok_base + l * batch] = sum_k;
                        }

                        for i in 0..d {
                            let mut sum_v = 0.0;
                            for l in 0..d {
                                sum_v += d_kv_local[i * d + l]
                                    * cache.k[tok_base + l * batch];
                            }
                            d_v[tok_base + i * batch] = sum_v;
                        }
                    }
                }

                // ================== 4. d_q_raw, d_k_raw, gi, grad параметров Q, K, V ==================
                // Пересчитываем q_raw, k_raw и применяем производные φ.
                for r in 0..batch {
                    for t in 0..seq {
                        let tok_base = (t * d) * batch + r;

                        for j in 0..d {
                            // Пересчёт q_raw[r,t,j], k_raw[r,t,j].
                            let mut q_raw_val = p[bq_start + j];
                            let mut k_raw_val = p[bk_start + j];
                            for i in 0..d {
                                let xv = x[(t * d + i) * batch + r];
                                q_raw_val += xv * p[wq_start + j * d + i];
                                k_raw_val += xv * p[wk_start + j * d + i];
                            }

                            let dq = d_q_phi[tok_base + j * batch]
                                * phi_derivative(q_raw_val);
                            let dk = d_k_phi[tok_base + j * batch]
                                * phi_derivative(k_raw_val);
                            let dv = d_v[tok_base + j * batch];

                            grad_bq[j] += dq;
                            grad_bk[j] += dk;
                            grad_bv[j] += dv;

                            for i in 0..d {
                                let xv = x[(t * d + i) * batch + r];

                                grad_wq[j * d + i] += dq * xv;
                                grad_wk[j * d + i] += dk * xv;
                                grad_wv[j * d + i] += dv * xv;

                                gi[tok_base + i * batch] += dq * p[wq_start + j * d + i]
                                    + dk * p[wk_start + j * d + i]
                                    + dv * p[wv_start + j * d + i];
                            }
                        }
                    }
                }

                // ================== 5. Запись градиентов ==================
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