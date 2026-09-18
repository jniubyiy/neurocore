// src/layers/per_feature_attention/cpu/mod.rs

use crate::compute_manager::core::dynamic_context::DynamicContext;
use crate::compute_manager::operators_v2::memory_v2::buffer::{MatrixBufferHandle, TempMatrixPool};
use crate::layers::buffered_context::{BufferedContext, CpuPerFeatureAttentionHead};
use crate::layers::UniversalLayerBuffered;
use crate::model_plan::param_store::ParamSlice;

use super::super::per_feature_attention::PerFeatureAttention;

// ============================================================================
// Математика
// ============================================================================

/// φ(x) = ELU(x) + 1 — точно так же, как в CPU-версии LinearAttention
/// (сохранена некоторая несогласованность на x=0, чтобы совпадать с уже
/// существующей CPU/GPU парой в проекте).
#[inline]
fn phi(x: f32) -> f32 {
    if x > 0.0 { x + 1.0 } else { x.exp() + 1.0 }
}

#[inline]
fn phi_derivative(x: f32) -> f32 {
    if x > 0.0 { 1.0 } else { x.exp() }
}

// ============================================================================
// Forward
// ============================================================================

impl UniversalLayerBuffered for PerFeatureAttention {
    fn forward_buffered(
        &self,
        input: &MatrixBufferHandle,
        output: &MatrixBufferHandle,
        params: &MatrixBufferHandle,
        slice: &ParamSlice,
        pool: &mut TempMatrixPool,
    ) -> BufferedContext {
        let batch = input.rows();
        let seq_len = self.seq_len;
        let d_model = self.d_model;
        let d_head = self.d_head;
        let head_pc = self.head_param_count();

        debug_assert_eq!(input.cols(), seq_len * d_model);
        debug_assert_eq!(output.rows(), batch);
        debug_assert_eq!(output.cols(), seq_len * d_model);
        debug_assert!(
            slice.start + self.param_len() <= params.rows() * params.cols(),
            "PerFeatureAttention: parameter slice out of bounds"
        );

        let total_ts = batch * seq_len;

        // Читаем весь блок параметров слоя.
        let param_vec = params.read_range(slice.start, self.param_len());

        // ------------------------------------------------------------------
        // Аллокация per-head state-буферов. Раскладка column-major:
        //
        //   x_buf, dx_buf          : (batch · seq_len, 1)
        //                            индекс = t · batch + r
        //   q_raw, k_raw, v_raw,
        //   q_phi, k_phi, attn     : (batch · seq_len · d_head, 1)
        //                            индекс = (t · d_head + i) · batch + r
        //   kv                     : (batch · d_head · d_head, 1)
        //                            индекс = (i · d_head + j) · batch + r
        //   z                      : (batch · d_head, 1)
        //                            индекс = i · batch + r
        // ------------------------------------------------------------------
        let mut cpu_heads: Vec<CpuPerFeatureAttentionHead> = Vec::with_capacity(d_model);
        for h in 0..d_model {
            let x_buf = pool.acquire(total_ts, 1);
            let dx_buf = pool.acquire(total_ts, 1);
            let q_raw_buf = pool.acquire(total_ts * d_head, 1);
            let k_raw_buf = pool.acquire(total_ts * d_head, 1);
            let v_raw_buf = pool.acquire(total_ts * d_head, 1);
            let q_phi_buf = pool.acquire(total_ts * d_head, 1);
            let k_phi_buf = pool.acquire(total_ts * d_head, 1);
            let kv_buf = pool.acquire(batch * d_head * d_head, 1);
            let z_buf = pool.acquire(batch * d_head, 1);
            let attn_buf = pool.acquire(total_ts * d_head, 1);

            let head_base = h * head_pc;
            let self_bias_off = head_base + 10 * d_head + 1;
            let self_bias = param_vec[self_bias_off];

            cpu_heads.push(CpuPerFeatureAttentionHead {
                x: x_buf,
                dx: dx_buf,
                q_raw: q_raw_buf,
                k_raw: k_raw_buf,
                v_raw: v_raw_buf,
                q_phi: q_phi_buf,
                k_phi: k_phi_buf,
                kv: kv_buf,
                z: z_buf,
                attn: attn_buf,
                self_bias,
                head_index: h,
            });
        }

        // ------------------------------------------------------------------
        // Основной проход — по одной голове за раз.
        // ------------------------------------------------------------------
        for h in 0..d_model {
            let head_base = h * head_pc;
            let wq_off = head_base;
            let bq_off = wq_off + 2 * d_head;
            let wk_off = bq_off + d_head;
            let bk_off = wk_off + 2 * d_head;
            let wv_off = bk_off + d_head;
            let bv_off = wv_off + 2 * d_head;
            let wo_off = bv_off + d_head;
            let bo_off = wo_off + d_head;

            let hc = &cpu_heads[h];
            let sb = hc.self_bias;

            let ids = [
                input.id(),
                output.id(),
                hc.x.id(),
                hc.dx.id(),
                hc.q_raw.id(),
                hc.k_raw.id(),
                hc.v_raw.id(),
                hc.q_phi.id(),
                hc.k_phi.id(),
                hc.kv.id(),
                hc.z.id(),
                hc.attn.id(),
            ];

            input
                .memory()
                .write()
                .unwrap()
                .with_cpu_slices_mut(&ids, |slices| {
                    let (first, rest) = slices.split_at_mut(1);
                    let in_slice: &[f32] = &*first[0];

                    let (second, rest) = rest.split_at_mut(1);
                    let out_slice: &mut [f32] = &mut *second[0];

                    let (third, rest) = rest.split_at_mut(1);
                    let x_slice: &mut [f32] = &mut *third[0];

                    let (fourth, rest) = rest.split_at_mut(1);
                    let dx_slice: &mut [f32] = &mut *fourth[0];

                    let (fifth, rest) = rest.split_at_mut(1);
                    let q_raw_slice: &mut [f32] = &mut *fifth[0];

                    let (sixth, rest) = rest.split_at_mut(1);
                    let k_raw_slice: &mut [f32] = &mut *sixth[0];

                    let (seventh, rest) = rest.split_at_mut(1);
                    let v_raw_slice: &mut [f32] = &mut *seventh[0];

                    let (eighth, rest) = rest.split_at_mut(1);
                    let q_phi_slice: &mut [f32] = &mut *eighth[0];

                    let (ninth, rest) = rest.split_at_mut(1);
                    let k_phi_slice: &mut [f32] = &mut *ninth[0];

                    let (tenth, rest) = rest.split_at_mut(1);
                    let kv_slice: &mut [f32] = &mut *tenth[0];

                    let (eleventh, rest) = rest.split_at_mut(1);
                    let z_slice: &mut [f32] = &mut *eleventh[0];

                    let (twelfth, _) = rest.split_at_mut(1);
                    let attn_slice: &mut [f32] = &mut *twelfth[0];

                    // 1. Извлечение x и вычисление dx.
                    for r in 0..batch {
                        for t in 0..seq_len {
                            let src = (t * d_model + h) * batch + r;
                            let dst = t * batch + r;
                            x_slice[dst] = in_slice[src];
                        }
                    }
                    for r in 0..batch {
                        for t in 0..seq_len {
                            let dst = t * batch + r;
                            if t == 0 {
                                dx_slice[dst] = x_slice[dst];
                            } else {
                                let prev = (t - 1) * batch + r;
                                dx_slice[dst] = x_slice[dst] - x_slice[prev];
                            }
                        }
                    }

                    // 2. QKV-проекции и φ.
                    for r in 0..batch {
                        for t in 0..seq_len {
                            let ts_idx = t * batch + r;
                            let x_val = x_slice[ts_idx];
                            let dx_val = dx_slice[ts_idx];

                            for i in 0..d_head {
                                let q_w0 = param_vec[wq_off + i * 2];
                                let q_w1 = param_vec[wq_off + i * 2 + 1];
                                let q_b = param_vec[bq_off + i];
                                let q_val = q_w0 * x_val + q_w1 * dx_val + q_b;

                                let k_w0 = param_vec[wk_off + i * 2];
                                let k_w1 = param_vec[wk_off + i * 2 + 1];
                                let k_b = param_vec[bk_off + i];
                                let k_val = k_w0 * x_val + k_w1 * dx_val + k_b;

                                let v_w0 = param_vec[wv_off + i * 2];
                                let v_w1 = param_vec[wv_off + i * 2 + 1];
                                let v_b = param_vec[bv_off + i];
                                let v_val = v_w0 * x_val + v_w1 * dx_val + v_b;

                                let idx = (t * d_head + i) * batch + r;
                                q_raw_slice[idx] = q_val;
                                k_raw_slice[idx] = k_val;
                                v_raw_slice[idx] = v_val;
                                q_phi_slice[idx] = phi(q_val);
                                k_phi_slice[idx] = phi(k_val);
                            }
                        }
                    }

                    // 3. kv и z (per-example).
                    for r in 0..batch {
                        for i in 0..d_head {
                            for j in 0..d_head {
                                let mut acc = 0.0f32;
                                for t in 0..seq_len {
                                    let ki = (t * d_head + i) * batch + r;
                                    let vj = (t * d_head + j) * batch + r;
                                    acc += k_phi_slice[ki] * v_raw_slice[vj];
                                }
                                let kv_idx = (i * d_head + j) * batch + r;
                                kv_slice[kv_idx] = acc;
                            }
                        }
                        for i in 0..d_head {
                            let mut acc = 0.0f32;
                            for t in 0..seq_len {
                                let ki = (t * d_head + i) * batch + r;
                                acc += k_phi_slice[ki];
                            }
                            z_slice[i * batch + r] = acc;
                        }
                    }

                    // 4. Attention.
                    for r in 0..batch {
                        for t in 0..seq_len {
                            let base = (t * d_head) * batch + r;

                            let mut denom = sb;
                            for i in 0..d_head {
                                let qp = q_phi_slice[base + i * batch];
                                let zi = z_slice[i * batch + r];
                                denom += qp * zi;
                            }
                            let inv_denom = 1.0 / denom;

                            for j in 0..d_head {
                                let v_t_j = v_raw_slice[base + j * batch];
                                let mut num_j = sb * v_t_j;
                                for i in 0..d_head {
                                    let qp = q_phi_slice[base + i * batch];
                                    let kv_ij = kv_slice[(i * d_head + j) * batch + r];
                                    num_j += qp * kv_ij;
                                }
                                attn_slice[base + j * batch] = num_j * inv_denom;
                            }
                        }
                    }

                    // 5. Выходная проекция.
                    for r in 0..batch {
                        for t in 0..seq_len {
                            let base = (t * d_head) * batch + r;
                            let mut y = param_vec[bo_off];
                            for j in 0..d_head {
                                let wo_j = param_vec[wo_off + j];
                                y += wo_j * attn_slice[base + j * batch];
                            }
                            let dst = (t * d_model + h) * batch + r;
                            out_slice[dst] = y;
                        }
                    }
                });
        }

        BufferedContext::PerFeatureAttention {
            input: input.clone(),
            cpu_heads,
            batch,
            seq_len,
            d_model,
            d_head,
        }
    }

    // ========================================================================
    // Backward
    // ========================================================================

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
        let (input_handle, cpu_heads, batch, seq_len, d_model, d_head) = match bc {
            BufferedContext::PerFeatureAttention {
                input,
                cpu_heads,
                batch,
                seq_len,
                d_model,
                d_head,
            } => (
                input,
                cpu_heads,
                *batch,
                *seq_len,
                *d_model,
                *d_head,
            ),
            _ => panic!("Expected PerFeatureAttention Buffered context"),
        };

        let head_pc = self.head_param_count();
        let total_ts = batch * seq_len;

        debug_assert_eq!(grad_output.rows(), batch);
        debug_assert_eq!(grad_output.cols(), seq_len * d_model);
        debug_assert_eq!(grad_input.rows(), batch);
        debug_assert_eq!(grad_input.cols(), seq_len * d_model);
        debug_assert_eq!(input_handle.rows(), batch);
        debug_assert_eq!(input_handle.cols(), seq_len * d_model);

        // Читаем параметры и grad_output.
        let param_vec = params.read_range(slice.start, self.param_len());
        let go_vec = grad_output.read_range(0, batch * seq_len * d_model);

        // Локальный накопитель градиентов параметров слоя (та же длина,
        // что param_len). В конце запишем одним write_range.
        let mut grad_params_local = vec![0.0f32; self.param_len()];

        // Градиент по входу. Накапливаем в плоский (batch · seq_len · d_model).
        let mut grad_x_global = vec![0.0f32; batch * seq_len * d_model];

        for h in 0..d_model {
            let head_base = h * head_pc;
            let wq_off = head_base;
            let bq_off = wq_off + 2 * d_head;
            let wk_off = bq_off + d_head;
            let bk_off = wk_off + 2 * d_head;
            let wv_off = bk_off + d_head;
            let bv_off = wv_off + 2 * d_head;
            let wo_off = bv_off + d_head;
            let bo_off = wo_off + d_head;
            let self_bias_off = bo_off + 1;

            let hc = &cpu_heads[h];
            let self_bias = hc.self_bias;

            // Читаем state.
            let x_vec = hc.x.read_range(0, total_ts);
            let dx_vec = hc.dx.read_range(0, total_ts);
            let q_raw_vec = hc.q_raw.read_range(0, total_ts * d_head);
            let k_raw_vec = hc.k_raw.read_range(0, total_ts * d_head);
            let v_raw_vec = hc.v_raw.read_range(0, total_ts * d_head);
            let q_phi_vec = hc.q_phi.read_range(0, total_ts * d_head);
            let k_phi_vec = hc.k_phi.read_range(0, total_ts * d_head);
            let kv_vec = hc.kv.read_range(0, batch * d_head * d_head);
            let z_vec = hc.z.read_range(0, batch * d_head);
            let attn_vec = hc.attn.read_range(0, total_ts * d_head);

            // Локальные аккумуляторы.
            let mut grad_x_local = vec![0.0f32; total_ts];
            let mut grad_q_raw = vec![0.0f32; total_ts * d_head];
            let mut grad_k_raw = vec![0.0f32; total_ts * d_head];
            let mut grad_v_raw = vec![0.0f32; total_ts * d_head];
            let mut grad_kv = vec![0.0f32; batch * d_head * d_head];
            let mut grad_z = vec![0.0f32; batch * d_head];
            let mut grad_q_phi = vec![0.0f32; total_ts * d_head];
            let mut grad_self_bias = 0.0f32;

            // ---- 1. Backward через выходную проекцию. ----
            let mut grad_attn = vec![0.0f32; total_ts * d_head];
            for r in 0..batch {
                for t in 0..seq_len {
                    let out_idx = (t * d_model + h) * batch + r;
                    let go = go_vec[out_idx];

                    grad_params_local[bo_off] += go;

                    let base = (t * d_head) * batch + r;
                    for j in 0..d_head {
                        let wo_j = param_vec[wo_off + j];
                        grad_params_local[wo_off + j] += go * attn_vec[base + j * batch];
                        grad_attn[base + j * batch] = go * wo_j;
                    }
                }
            }

            // ---- 2. Backward через attention. ----
            for r in 0..batch {
                for t in 0..seq_len {
                    let base = (t * d_head) * batch + r;

                    // Пересчёт denom.
                    let mut denom = self_bias;
                    for i in 0..d_head {
                        denom += q_phi_vec[base + i * batch] * z_vec[i * batch + r];
                    }
                    let inv_denom = 1.0 / denom;

                    // dL/ddenom = −Σ_j dattn[j] · attn[j] / denom.
                    let mut ddenom = 0.0f32;
                    for j in 0..d_head {
                        let dattn_j = grad_attn[base + j * batch];
                        ddenom -= dattn_j * attn_vec[base + j * batch] * inv_denom;
                    }

                    // Backward через num.
                    for j in 0..d_head {
                        let dattn_j = grad_attn[base + j * batch];
                        let dnum_j = dattn_j * inv_denom;

                        grad_v_raw[base + j * batch] += self_bias * dnum_j;

                        for i in 0..d_head {
                            let qp = q_phi_vec[base + i * batch];
                            let kv_ij = kv_vec[(i * d_head + j) * batch + r];
                            grad_q_phi[base + i * batch] += kv_ij * dnum_j;
                            grad_kv[(i * d_head + j) * batch + r] += qp * dnum_j;
                        }

                        grad_self_bias += dnum_j * v_raw_vec[base + j * batch];
                    }

                    // Backward через denom.
                    grad_self_bias += ddenom;
                    for i in 0..d_head {
                        let qp = q_phi_vec[base + i * batch];
                        grad_q_phi[base + i * batch] += ddenom * z_vec[i * batch + r];
                        grad_z[i * batch + r] += ddenom * qp;
                    }
                }
            }

            // ---- 3. Backward через kv и z. ----
            for r in 0..batch {
                for t in 0..seq_len {
                    let base = (t * d_head) * batch + r;
                    for i in 0..d_head {
                        let mut dk_phi_i = grad_z[i * batch + r];
                        for j in 0..d_head {
                            let dkv_ij = grad_kv[(i * d_head + j) * batch + r];
                            dk_phi_i += dkv_ij * v_raw_vec[base + j * batch];
                            grad_v_raw[base + j * batch] += dkv_ij * k_phi_vec[base + i * batch];
                        }
                        let k_raw_val = k_raw_vec[base + i * batch];
                        grad_k_raw[base + i * batch] += dk_phi_i * phi_derivative(k_raw_val);
                    }
                }
            }

            // ---- 4. Backward через φ для q. ----
            for r in 0..batch {
                for t in 0..seq_len {
                    let base = (t * d_head) * batch + r;
                    for i in 0..d_head {
                        let q_raw_val = q_raw_vec[base + i * batch];
                        grad_q_raw[base + i * batch] =
                            grad_q_phi[base + i * batch] * phi_derivative(q_raw_val);
                    }
                }
            }

            // ---- 5. Backward через QKV-проекции. ----
            let mut grad_u = vec![0.0f32; total_ts * 2];
            for r in 0..batch {
                for t in 0..seq_len {
                    let ts_idx = t * batch + r;
                    let base = (t * d_head) * batch + r;

                    let x_val = x_vec[ts_idx];
                    let dx_val = dx_vec[ts_idx];

                    let mut du_0 = 0.0f32;
                    let mut du_1 = 0.0f32;

                    for i in 0..d_head {
                        let dq = grad_q_raw[base + i * batch];
                        let dk = grad_k_raw[base + i * batch];
                        let dv = grad_v_raw[base + i * batch];

                        let wq0 = param_vec[wq_off + i * 2];
                        let wq1 = param_vec[wq_off + i * 2 + 1];
                        let wk0 = param_vec[wk_off + i * 2];
                        let wk1 = param_vec[wk_off + i * 2 + 1];
                        let wv0 = param_vec[wv_off + i * 2];
                        let wv1 = param_vec[wv_off + i * 2 + 1];

                        grad_params_local[wq_off + i * 2] += dq * x_val;
                        grad_params_local[wq_off + i * 2 + 1] += dq * dx_val;
                        grad_params_local[bq_off + i] += dq;

                        grad_params_local[wk_off + i * 2] += dk * x_val;
                        grad_params_local[wk_off + i * 2 + 1] += dk * dx_val;
                        grad_params_local[bk_off + i] += dk;

                        grad_params_local[wv_off + i * 2] += dv * x_val;
                        grad_params_local[wv_off + i * 2 + 1] += dv * dx_val;
                        grad_params_local[bv_off + i] += dv;

                        du_0 += dq * wq0 + dk * wk0 + dv * wv0;
                        du_1 += dq * wq1 + dk * wk1 + dv * wv1;
                    }

                    grad_u[ts_idx * 2] = du_0;
                    grad_u[ts_idx * 2 + 1] = du_1;
                }
            }

            // ---- 6. Backward через u = [x, dx]. ----
            // dx[t] = x[t] − x[t−1], dx[0] = x[0].
            for r in 0..batch {
                for t in 0..seq_len {
                    let ts_idx = t * batch + r;
                    let du_0 = grad_u[ts_idx * 2];
                    let du_1 = grad_u[ts_idx * 2 + 1];

                    grad_x_local[ts_idx] += du_0 + du_1;

                    if t > 0 {
                        let prev = (t - 1) * batch + r;
                        grad_x_local[prev] -= du_1;
                    }
                }
            }

            // ---- 7. self_bias. ----
            grad_params_local[self_bias_off] += grad_self_bias;

            // ---- 8. Разложить grad_x_local в grad_x_global. ----
            for r in 0..batch {
                for t in 0..seq_len {
                    let ts_idx = t * batch + r;
                    let dst = (t * d_model + h) * batch + r;
                    grad_x_global[dst] = grad_x_local[ts_idx];
                }
            }
        }

        grad_params.write_range(slice.start, &grad_params_local);
        grad_input.write_range(0, &grad_x_global);
    }

    fn param_len(&self) -> usize {
        self.d_model * self.head_param_count()
    }

    fn input_features(&self) -> usize {
        self.seq_len * self.d_model
    }

    fn output_features(&self) -> usize {
        self.seq_len * self.d_model
    }
}