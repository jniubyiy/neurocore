// src/layers/linear_attention/cpu/mod.rs

use crate::compute_manager::graph::types::DynamicContext;
use crate::compute_manager::matrix_buffer::{MatrixBufferHandle, TempMatrixPool};
use crate::layers::buffered_context::{BufferedContext, CpuLinearAttentionHead};
use crate::layers::UniversalLayerBuffered;
use crate::model_plan::param_store::ParamSlice;

use super::super::linear_attention::linear_attention::LinearAttention;

// ============================================================================
// Константы «резинки с горкой»
// ============================================================================
//
// A (правка ядра): H_SOFT_TAU расширен с 0.1 до 0.5.
// D (правка ядра): HEAD_WEIGHT_K понижен с 8.0 до 4.0.
// B (правка ядра): порог θ_k = k − 1.0 (см. compute_h_soft).
//
// FIX: kv и z теперь per-example (по r), а не глобальные.

const H_SOFT_TAU: f32 = 0.5;
const HEAD_WEIGHT_K: f32 = 4.0;
const HEAD_WEIGHT_EPS: f32 = 1e-6;

// ============================================================================
// Математика
// ============================================================================

fn phi(x: f32) -> f32 {
    if x > 0.0 { x + 1.0 } else { x.exp() + 1.0 }
}

fn phi_derivative(x: f32) -> f32 {
    if x > 0.0 { 1.0 } else { x.exp() }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

#[inline]
fn compute_h_soft(h_raw: f32, min_heads: usize, max_heads: usize) -> f32 {
    if min_heads >= max_heads {
        return min_heads as f32;
    }
    let n_trans = max_heads - min_heads;
    let mut h = min_heads as f32;
    for k in 1..=n_trans {
        // B: θ_k = k − 1.0.
        let theta_k = (k as f32) - 1.0;
        h += sigmoid((h_raw - theta_k) / H_SOFT_TAU);
    }
    h
}

#[inline]
fn compute_h_soft_derivative(h_raw: f32, min_heads: usize, max_heads: usize) -> f32 {
    if min_heads >= max_heads {
        return 0.0;
    }
    let n_trans = max_heads - min_heads;
    let mut d = 0.0f32;
    for k in 1..=n_trans {
        // B: θ_k = k − 1.0.
        let theta_k = (k as f32) - 1.0;
        let s = sigmoid((h_raw - theta_k) / H_SOFT_TAU);
        d += s * (1.0 - s) / H_SOFT_TAU;
    }
    d
}

#[inline]
fn head_weight(h_soft: f32, h: usize) -> f32 {
    let x = h_soft - h as f32;
    0.5 * (1.0 + ((x - 0.5) * HEAD_WEIGHT_K).tanh())
}

#[inline]
fn head_weight_derivative(h_soft: f32, h: usize) -> f32 {
    let x = h_soft - h as f32;
    let t = ((x - 0.5) * HEAD_WEIGHT_K).tanh();
    0.5 * HEAD_WEIGHT_K * (1.0 - t * t)
}

// ============================================================================
// Прямой проход одной головы
// ============================================================================

struct HeadForwardResult {
    q_phi: Vec<f32>,
    k_phi: Vec<f32>,
    v_raw: Vec<f32>,
    /// FIX: kv — per-example, layout `kv[r * dh * dh + i * dh + l]`.
    /// Раньше был глобальный размер dh·dh.
    kv: Vec<f32>,
    /// FIX: z — per-example, layout `z[r * dh + l]`.
    /// Раньше был глобальный размер dh.
    z: Vec<f32>,
    attn_out: Vec<f32>,
    y_h: Vec<f32>,
}

fn compute_head_forward(
    x: &[f32],
    p: &[f32],
    batch: usize,
    seq: usize,
    d: usize,
    dh: usize,
    head_base: usize,
) -> HeadForwardResult {
    let total_in = batch * seq * d;
    let total_head = batch * seq * dh;

    let wq_start = head_base;
    let bq_start = wq_start + dh * d;
    let wk_start = bq_start + dh;
    let bk_start = wk_start + dh * d;
    let wv_start = bk_start + dh;
    let bv_start = wv_start + dh * d;
    let wo_start = bv_start + dh;
    let bo_start = wo_start + d * dh;
    let self_bias_idx = bo_start + d;
    let self_bias = p[self_bias_idx];

    // 1. QKV-проекции.
    let mut q_raw = vec![0.0f32; total_head];
    let mut k_raw = vec![0.0f32; total_head];
    let mut v_raw = vec![0.0f32; total_head];

    for r in 0..batch {
        for t in 0..seq {
            let x_base = (t * d) * batch + r;
            let h_base = (t * dh) * batch + r;
            for j in 0..dh {
                let mut sq = p[bq_start + j];
                let mut sk = p[bk_start + j];
                let mut sv = p[bv_start + j];
                for i in 0..d {
                    let xv = x[x_base + i * batch];
                    sq += xv * p[wq_start + j * d + i];
                    sk += xv * p[wk_start + j * d + i];
                    sv += xv * p[wv_start + j * d + i];
                }
                let idx = h_base + j * batch;
                q_raw[idx] = sq;
                k_raw[idx] = sk;
                v_raw[idx] = sv;
            }
        }
    }

    // 2. φ.
    let mut q_phi = vec![0.0f32; total_head];
    let mut k_phi = vec![0.0f32; total_head];
    for i in 0..total_head {
        q_phi[i] = phi(q_raw[i]);
        k_phi[i] = phi(k_raw[i]);
    }

    // 3. kv и z (FIX: per-example).
    //
    // Layout (column-major по внутренним осям):
    //   kv[r · dh·dh + i · dh + l] = Σ_t k_phi[r,t,l] · v_raw[r,t,i]
    //   z [r · dh + l]             = Σ_t k_phi[r,t,l]
    let mut kv = vec![0.0f32; batch * dh * dh];
    let mut z = vec![0.0f32; batch * dh];
    for r in 0..batch {
        let z_r = r * dh;
        let kv_r = r * dh * dh;
        for t in 0..seq {
            let h_base = (t * dh) * batch + r;
            for i in 0..dh {
                let ki = k_phi[h_base + i * batch];
                z[z_r + i] += ki;
                for j in 0..dh {
                    let vj = v_raw[h_base + j * batch];
                    kv[kv_r + j * dh + i] += ki * vj;
                }
            }
        }
    }

    // 4. attn_out (FIX: используем per-example kv и z, индекс r).
    let eps = 1e-6f32;
    let mut attn_out = vec![0.0f32; total_head];
    for r in 0..batch {
        let z_r = r * dh;
        let kv_r = r * dh * dh;
        for t in 0..seq {
            let h_base = (t * dh) * batch + r;
            let mut denom = eps + self_bias;
            for l in 0..dh {
                denom += q_phi[h_base + l * batch] * z[z_r + l];
            }
            let inv_denom = 1.0 / denom;
            for i in 0..dh {
                let mut num = self_bias * v_raw[h_base + i * batch];
                for l in 0..dh {
                    num += q_phi[h_base + l * batch] * kv[kv_r + i * dh + l];
                }
                attn_out[h_base + i * batch] = num * inv_denom;
            }
        }
    }

    // 5. Выходная проекция.
    let mut y_h = vec![0.0f32; total_in];
    for r in 0..batch {
        for t in 0..seq {
            let h_base = (t * dh) * batch + r;
            let y_base = (t * d) * batch + r;
            for k in 0..d {
                let mut sum = p[bo_start + k];
                for j in 0..dh {
                    sum += attn_out[h_base + j * batch] * p[wo_start + k * dh + j];
                }
                y_h[y_base + k * batch] = sum;
            }
        }
    }

    HeadForwardResult { q_phi, k_phi, v_raw, kv, z, attn_out, y_h }
}

// ============================================================================
// Обратный проход одной головы
// ============================================================================

struct HeadBackwardResult {
    gi: Vec<f32>,
    grad_wq: Vec<f32>,
    grad_bq: Vec<f32>,
    grad_wk: Vec<f32>,
    grad_bk: Vec<f32>,
    grad_wv: Vec<f32>,
    grad_bv: Vec<f32>,
    grad_wo: Vec<f32>,
    grad_bo: Vec<f32>,
    grad_self_bias: f32,
}

#[allow(clippy::too_many_arguments)]
fn compute_head_backward(
    x: &[f32],
    go_h: &[f32],
    p: &[f32],
    head_base: usize,
    q_phi: &[f32],
    k_phi: &[f32],
    v_raw: &[f32],
    kv: &[f32],          // FIX: теперь (batch · dh · dh)
    z: &[f32],           // FIX: теперь (batch · dh)
    attn_out: &[f32],
    batch: usize,
    seq: usize,
    d: usize,
    dh: usize,
) -> HeadBackwardResult {
    let total_in = batch * seq * d;
    let total_head = batch * seq * dh;

    let wq_start = head_base;
    let bq_start = wq_start + dh * d;
    let wk_start = bq_start + dh;
    let bk_start = wk_start + dh * d;
    let wv_start = bk_start + dh;
    let bv_start = wv_start + dh * d;
    let wo_start = bv_start + dh;
    let bo_start = wo_start + d * dh;
    let self_bias_idx = bo_start + d;
    let self_bias = p[self_bias_idx];

    let mut grad_wq = vec![0.0f32; dh * d];
    let mut grad_bq = vec![0.0f32; dh];
    let mut grad_wk = vec![0.0f32; dh * d];
    let mut grad_bk = vec![0.0f32; dh];
    let mut grad_wv = vec![0.0f32; dh * d];
    let mut grad_bv = vec![0.0f32; dh];
    let mut grad_wo = vec![0.0f32; d * dh];
    let mut grad_bo = vec![0.0f32; d];

    // 1. d_attn_out, grad_Wo, grad_bo.
    let mut d_attn_out = vec![0.0f32; total_head];
    for r in 0..batch {
        for t in 0..seq {
            let h_base = (t * dh) * batch + r;
            let y_base = (t * d) * batch + r;

            for k in 0..d {
                let go_k = go_h[y_base + k * batch];
                grad_bo[k] += go_k;
                for j in 0..dh {
                    grad_wo[k * dh + j] += go_k * attn_out[h_base + j * batch];
                }
            }
            for j in 0..dh {
                let mut sum = 0.0f32;
                for k in 0..d {
                    sum += go_h[y_base + k * batch] * p[wo_start + k * dh + j];
                }
                d_attn_out[h_base + j * batch] = sum;
            }
        }
    }

    // 2. d_q_phi, d_kv, d_z, d_self_bias.
    //
    // FIX: d_kv и d_z теперь per-example:
    //   d_kv[r · dh·dh + i · dh + l]
    //   d_z [r · dh + l]
    let eps = 1e-6f32;
    let mut d_q_phi = vec![0.0f32; total_head];
    let mut d_kv = vec![0.0f32; batch * dh * dh];
    let mut d_z = vec![0.0f32; batch * dh];
    let mut d_self_bias = 0.0f32;
    let mut inv_denoms = vec![0.0f32; batch * seq];

    for r in 0..batch {
        let z_r = r * dh;
        let kv_r = r * dh * dh;
        let d_kv_r = r * dh * dh;
        let d_z_r = r * dh;

        for t in 0..seq {
            let h_base = (t * dh) * batch + r;
            let mut denom = eps + self_bias;
            for l in 0..dh {
                denom += q_phi[h_base + l * batch] * z[z_r + l];
            }
            let inv_denom = 1.0 / denom;
            let inv_denom_sq = inv_denom * inv_denom;
            inv_denoms[t * batch + r] = inv_denom;

            let mut sum_da_ao = 0.0f32;
            for i in 0..dh {
                let da = d_attn_out[h_base + i * batch];
                let ao = attn_out[h_base + i * batch];
                sum_da_ao += da * ao;
            }

            for l in 0..dh {
                let q_l = q_phi[h_base + l * batch];
                let mut dq_l = 0.0;
                for i in 0..dh {
                    let da = d_attn_out[h_base + i * batch];
                    // FIX: kv[r, i, l] — per-example.
                    let kv_il = kv[kv_r + i * dh + l];
                    dq_l += da * kv_il * inv_denom;
                    // FIX: d_kv[r, i, l] — per-example.
                    d_kv[d_kv_r + i * dh + l] += da * q_l * inv_denom;
                }
                dq_l -= sum_da_ao * z[z_r + l] * inv_denom_sq;
                d_q_phi[h_base + l * batch] = dq_l;
                // FIX: d_z[r, l] — per-example.
                d_z[d_z_r + l] += -q_l * inv_denom * sum_da_ao;
            }

            d_self_bias += -inv_denom * sum_da_ao;
            for i in 0..dh {
                let da = d_attn_out[h_base + i * batch];
                let v_val = v_raw[h_base + i * batch];
                d_self_bias += da * inv_denom * v_val;
            }
        }
    }

    // 3. d_k_phi, d_v.
    let mut d_k_phi = vec![0.0f32; total_head];
    let mut d_v = vec![0.0f32; total_head];
    for r in 0..batch {
        let d_kv_r = r * dh * dh;
        let d_z_r = r * dh;
        for t in 0..seq {
            let h_base = (t * dh) * batch + r;
            let inv_denom = inv_denoms[t * batch + r];
            for l in 0..dh {
                let mut sum_k = d_z[d_z_r + l];
                for i in 0..dh {
                    sum_k += d_kv[d_kv_r + i * dh + l] * v_raw[h_base + i * batch];
                }
                d_k_phi[h_base + l * batch] = sum_k;
            }
            for i in 0..dh {
                let mut sum_v = 0.0f32;
                for l in 0..dh {
                    sum_v += d_kv[d_kv_r + i * dh + l] * k_phi[h_base + l * batch];
                }
                let da = d_attn_out[h_base + i * batch];
                sum_v += da * inv_denom * self_bias;
                d_v[h_base + i * batch] = sum_v;
            }
        }
    }

    // 4. gi, grad Wq/Wk/Wv, bq/bk/bv.
    let mut gi = vec![0.0f32; total_in];
    for r in 0..batch {
        for t in 0..seq {
            let h_base = (t * dh) * batch + r;
            let x_base = (t * d) * batch + r;

            for j in 0..dh {
                // Пересчёт q_raw, k_raw.
                let mut q_raw_val = p[bq_start + j];
                let mut k_raw_val = p[bk_start + j];
                for i in 0..d {
                    let xv = x[x_base + i * batch];
                    q_raw_val += xv * p[wq_start + j * d + i];
                    k_raw_val += xv * p[wk_start + j * d + i];
                }
                let dq = d_q_phi[h_base + j * batch] * phi_derivative(q_raw_val);
                let dk = d_k_phi[h_base + j * batch] * phi_derivative(k_raw_val);
                let dv = d_v[h_base + j * batch];

                grad_bq[j] += dq;
                grad_bk[j] += dk;
                grad_bv[j] += dv;

                for i in 0..d {
                    let xv = x[x_base + i * batch];
                    grad_wq[j * d + i] += dq * xv;
                    grad_wk[j * d + i] += dk * xv;
                    grad_wv[j * d + i] += dv * xv;
                    gi[x_base + i * batch] +=
                        dq * p[wq_start + j * d + i]
                        + dk * p[wk_start + j * d + i]
                        + dv * p[wv_start + j * d + i];
                }
            }
        }
    }

    HeadBackwardResult {
        gi,
        grad_wq,
        grad_bq,
        grad_wk,
        grad_bk,
        grad_wv,
        grad_bv,
        grad_wo,
        grad_bo,
        grad_self_bias: d_self_bias,
    }
}

// ============================================================================
// Forward
// ============================================================================

impl UniversalLayerBuffered for LinearAttention {
    fn forward_buffered(
        &self,
        input: &MatrixBufferHandle,
        output: &MatrixBufferHandle,
        params: &MatrixBufferHandle,
        slice: &ParamSlice,
        pool: &mut TempMatrixPool,
    ) -> BufferedContext {
        let batch = input.rows();
        let features = input.cols();
        let d = self.d_model;
        let seq = self.seq_len;
        let dh = self.d_head();
        let total_in = batch * seq * d;
        let total_head = batch * seq * dh;
        // FIX: размеры per-example state-буферов.
        let kv_per_head = batch * dh * dh;
        let z_per_head = batch * dh;

        debug_assert_eq!(features, seq * d);
        debug_assert_eq!(output.rows(), batch);
        debug_assert_eq!(output.cols(), seq * d);
        debug_assert!(slice.start + self.param_len() <= params.rows() * params.cols());

        let head_size = self.head_param_count();
        let h_raw_idx = slice.start + self.max_heads * head_size;

        let param_vec = params.read_range(slice.start, self.param_len());
        let x_vec = input.read_range(0, total_in);

        let mut head_buffers: Vec<CpuLinearAttentionHead> = Vec::new();
        let mut y_accum = vec![0.0f32; total_in];

        let h_raw = param_vec[h_raw_idx - slice.start];
        let h_soft = compute_h_soft(h_raw, self.min_heads, self.max_heads);

        for h in 0..self.max_heads {
            let w_h = head_weight(h_soft, h);
            if w_h <= HEAD_WEIGHT_EPS {
                continue;
            }

            let head_base = h * head_size;
            let res = compute_head_forward(
                &x_vec,
                &param_vec,
                batch,
                seq,
                d,
                dh,
                head_base,
            );

            for i in 0..total_in {
                y_accum[i] += w_h * res.y_h[i];
            }

            // FIX: буферы kv и z теперь per-example.
            let q_buf = pool.acquire(total_head, 1);
            let k_buf = pool.acquire(total_head, 1);
            let v_buf = pool.acquire(total_head, 1);
            let kv_buf = pool.acquire(kv_per_head, 1);
            let z_buf = pool.acquire(z_per_head, 1);
            let attn_buf = pool.acquire(total_head, 1);
            let y_buf = pool.acquire(total_in, 1);

            q_buf.write_range(0, &res.q_phi);
            k_buf.write_range(0, &res.k_phi);
            v_buf.write_range(0, &res.v_raw);
            kv_buf.write_range(0, &res.kv);
            z_buf.write_range(0, &res.z);
            attn_buf.write_range(0, &res.attn_out);
            y_buf.write_range(0, &res.y_h);

            head_buffers.push(CpuLinearAttentionHead {
                q: q_buf,
                k: k_buf,
                v: v_buf,
                kv: kv_buf,
                z: z_buf,
                attn_out: attn_buf,
                y: y_buf,
                weight: w_h,
                head_index: h,
            });
        }

        output.write_range(0, &y_accum);

        BufferedContext::LinearAttention {
            input: input.clone(),
            cpu_heads: head_buffers,
            h_raw,
            h_soft,
            batch,
            seq,
            d_model: d,
            d_head: dh,
            q_raw: None,
            k_raw: None,
            v_raw: None,
            q_phi: None,
            k_phi: None,
            kv: None,
            z: None,
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
        let (input_handle, cpu_heads, h_raw, h_soft, cached_batch, cached_seq, cached_d, cached_dh) =
            match bc {
                BufferedContext::LinearAttention {
                    input,
                    cpu_heads,
                    h_raw,
                    h_soft,
                    batch,
                    seq,
                    d_model,
                    d_head,
                    ..
                } => (input, cpu_heads, *h_raw, *h_soft, *batch, *seq, *d_model, *d_head),
                _ => panic!("Expected LinearAttention Buffered context"),
            };

        let batch = grad_output.rows();
        let seq = self.seq_len;
        let d = self.d_model;
        let dh = self.d_head();
        let total_in = batch * seq * d;

        debug_assert_eq!(grad_output.cols(), seq * d);
        debug_assert_eq!(grad_input.rows(), batch);
        debug_assert_eq!(grad_input.cols(), seq * d);
        debug_assert_eq!(batch, cached_batch);
        debug_assert_eq!(seq, cached_seq);
        debug_assert_eq!(d, cached_d);
        debug_assert_eq!(dh, cached_dh);
        debug_assert!(slice.start + self.param_len() <= params.rows() * params.cols());
        debug_assert!(slice.start + self.param_len() <= grad_params.rows() * grad_params.cols());

        let head_size = self.head_param_count();
        let h_raw_idx = slice.start + self.max_heads * head_size;

        struct HeadSnapshot {
            weight: f32,
            head_index: usize,
            q_phi: Vec<f32>,
            k_phi: Vec<f32>,
            v_raw: Vec<f32>,
            kv: Vec<f32>,          // FIX: batch · dh · dh
            z: Vec<f32>,           // FIX: batch · dh
            attn_out: Vec<f32>,
            y_h: Vec<f32>,
        }

        let total_head = batch * seq * dh;
        let kv_per_head = batch * dh * dh;
        let z_per_head = batch * dh;

        let mut snapshots: Vec<HeadSnapshot> = Vec::with_capacity(cpu_heads.len());
        for hc in cpu_heads {
            let q_phi = hc.q.read_range(0, total_head);
            let k_phi = hc.k.read_range(0, total_head);
            let v_raw = hc.v.read_range(0, total_head);
            // FIX: per-example kv и z.
            let kv = hc.kv.read_range(0, kv_per_head);
            let z = hc.z.read_range(0, z_per_head);
            let attn_out = hc.attn_out.read_range(0, total_head);
            let y_h = hc.y.read_range(0, total_in);

            snapshots.push(HeadSnapshot {
                weight: hc.weight,
                head_index: hc.head_index,
                q_phi,
                k_phi,
                v_raw,
                kv,
                z,
                attn_out,
                y_h,
            });
        }

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

                let (fifth, _) = rest.split_at_mut(1);
                let gp: &mut [f32] = &mut *fifth[0];

                for i in 0..self.param_len() {
                    gp[slice.start + i] = 0.0;
                }
                for v in gi.iter_mut() { *v = 0.0; }

                let mut d_l_dh_raw_total = 0.0f32;
                let mut go_h = vec![0.0f32; total_in];

                for snap in &snapshots {
                    let h = snap.head_index;
                    let w_h = snap.weight;
                    let head_base = slice.start + h * head_size;

                    for i in 0..total_in {
                        go_h[i] = w_h * go[i];
                    }

                    let mut d_l_dw_h = 0.0f32;
                    for i in 0..total_in {
                        d_l_dw_h += go[i] * snap.y_h[i];
                    }

                    let dw_dhs = head_weight_derivative(h_soft, h);
                    d_l_dh_raw_total += d_l_dw_h * dw_dhs;

                    let res = compute_head_backward(
                        x, &go_h, p, head_base,
                        &snap.q_phi, &snap.k_phi, &snap.v_raw,
                        &snap.kv, &snap.z, &snap.attn_out,
                        batch, seq, d, dh,
                    );

                    for i in 0..total_in {
                        gi[i] += res.gi[i];
                    }

                    let wq_start = head_base;
                    let bq_start = wq_start + dh * d;
                    let wk_start = bq_start + dh;
                    let bk_start = wk_start + dh * d;
                    let wv_start = bk_start + dh;
                    let bv_start = wv_start + dh * d;
                    let wo_start = bv_start + dh;
                    let bo_start = wo_start + d * dh;
                    let self_bias_idx = bo_start + d;

                    for i in 0..dh {
                        gp[bq_start + i] = res.grad_bq[i];
                        gp[bk_start + i] = res.grad_bk[i];
                        gp[bv_start + i] = res.grad_bv[i];
                    }
                    for i in 0..dh * d {
                        gp[wq_start + i] = res.grad_wq[i];
                        gp[wk_start + i] = res.grad_wk[i];
                        gp[wv_start + i] = res.grad_wv[i];
                    }
                    for i in 0..d {
                        gp[bo_start + i] = res.grad_bo[i];
                    }
                    for i in 0..d * dh {
                        gp[wo_start + i] = res.grad_wo[i];
                    }
                    gp[self_bias_idx] = res.grad_self_bias;
                }

                let dh_soft_dh_raw = compute_h_soft_derivative(h_raw, self.min_heads, self.max_heads);
                let d_l_dh_raw = d_l_dh_raw_total * dh_soft_dh_raw;
                gp[h_raw_idx] = d_l_dh_raw;
            });
    }

    fn param_len(&self) -> usize {
        self.max_heads * self.head_param_count() + 1
    }

    fn input_features(&self) -> usize {
        self.seq_len * self.d_model
    }

    fn output_features(&self) -> usize {
        self.seq_len * self.d_model
    }
}