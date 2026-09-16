// src/layers/linear_attention/cpu/mod.rs

use std::sync::atomic::{AtomicUsize, Ordering};

use once_cell::sync::Lazy;

use crate::compute_manager::graph::types::DynamicContext;
use crate::compute_manager::matrix_buffer::{MatrixBufferHandle, TempMatrixPool};
use crate::layers::buffered_context::{BufferedContext, CpuLinearAttentionHead};
use crate::layers::UniversalLayerBuffered;
use crate::model_plan::param_store::ParamSlice;

use super::super::linear_attention::linear_attention::LinearAttention;

// ============================================================================
// Константы «резинки с горкой»
// ============================================================================

const H_SOFT_TAU: f32 = 0.1;
const HEAD_WEIGHT_K: f32 = 8.0;
const HEAD_WEIGHT_EPS: f32 = 1e-6;

// ============================================================================
// Отладочная инфраструктура
// ============================================================================

static LINATT_DEBUG: Lazy<bool> =
    Lazy::new(|| std::env::var("NEUROCORE_DEBUG_LINATT").is_ok());
static LINATT_FWD_CALLS: Lazy<AtomicUsize> = Lazy::new(|| AtomicUsize::new(0));
static LINATT_BWD_CALLS: Lazy<AtomicUsize> = Lazy::new(|| AtomicUsize::new(0));

const LINATT_FWD_LOG_LIMIT: usize = 3;
const LINATT_BWD_LOG_LIMIT: usize = 3;
const LINATT_TRAJECTORY_EVERY: usize = 50;

fn linatt_l2(data: &[f32]) -> f64 {
    let mut s = 0.0f64;
    for &v in data {
        if v.is_finite() {
            s += (v as f64) * (v as f64);
        }
    }
    s.sqrt()
}

fn linatt_stats(name: &str, data: &[f32]) {
    if data.is_empty() {
        println!("    [LINATT] {}: <empty>", name);
        return;
    }
    let mut mn = f32::INFINITY;
    let mut mx = f32::NEG_INFINITY;
    let mut sum = 0.0f64;
    let mut nan_cnt = 0usize;
    let mut inf_cnt = 0usize;
    for &v in data {
        if v.is_nan() { nan_cnt += 1; continue; }
        if v.is_infinite() { inf_cnt += 1; continue; }
        if v < mn { mn = v; }
        if v > mx { mx = v; }
        sum += v as f64;
    }
    let finite = data.len().saturating_sub(nan_cnt + inf_cnt);
    let mean = if finite > 0 { sum / finite as f64 } else { 0.0 };
    println!(
        "    [LINATT] {}: len={}, min={:.6}, max={:.6}, mean={:.6}, l2={:.6}, nan={}, inf={}",
        name, data.len(), mn, mx, mean, linatt_l2(data), nan_cnt, inf_cnt
    );
}

fn linatt_first(label: &str, data: &[f32], k: usize) {
    let show = data.len().min(k);
    println!("    [LINATT] {} (first {}): {:?}", label, show, &data[..show]);
}

fn linatt_has_bad(data: &[f32]) -> bool {
    data.iter().any(|v| !v.is_finite())
}

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
        let theta_k = (k as f32) - 0.5;
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
        let theta_k = (k as f32) - 0.5;
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
    kv: Vec<f32>,
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

    // 3. kv и z.
    let mut kv = vec![0.0f32; dh * dh];
    let mut z = vec![0.0f32; dh];
    for r in 0..batch {
        for t in 0..seq {
            let h_base = (t * dh) * batch + r;
            for i in 0..dh {
                let ki = k_phi[h_base + i * batch];
                z[i] += ki;
                for j in 0..dh {
                    let vj = v_raw[h_base + j * batch];
                    kv[j * dh + i] += ki * vj;
                }
            }
        }
    }

    // 4. attn_out.
    let eps = 1e-6f32;
    let mut attn_out = vec![0.0f32; total_head];
    for r in 0..batch {
        for t in 0..seq {
            let h_base = (t * dh) * batch + r;
            let mut denom = eps + self_bias;
            for l in 0..dh {
                denom += q_phi[h_base + l * batch] * z[l];
            }
            let inv_denom = 1.0 / denom;
            for i in 0..dh {
                let mut num = self_bias * v_raw[h_base + i * batch];
                for l in 0..dh {
                    num += q_phi[h_base + l * batch] * kv[i * dh + l];
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
    kv: &[f32],
    z: &[f32],
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
    let eps = 1e-6f32;
    let mut d_q_phi = vec![0.0f32; total_head];
    let mut d_kv = vec![0.0f32; dh * dh];
    let mut d_z = vec![0.0f32; dh];
    let mut d_self_bias = 0.0f32;
    let mut inv_denoms = vec![0.0f32; batch * seq];

    for r in 0..batch {
        for t in 0..seq {
            let h_base = (t * dh) * batch + r;
            let mut denom = eps + self_bias;
            for l in 0..dh {
                denom += q_phi[h_base + l * batch] * z[l];
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
                    let kv_li = kv[i * dh + l];
                    dq_l += da * kv_li * inv_denom;
                    d_kv[i * dh + l] += da * q_l * inv_denom;
                }
                dq_l -= sum_da_ao * z[l] * inv_denom_sq;
                d_q_phi[h_base + l * batch] = dq_l;
                d_z[l] += -q_l * inv_denom * sum_da_ao;
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
        for t in 0..seq {
            let h_base = (t * dh) * batch + r;
            let inv_denom = inv_denoms[t * batch + r];
            for l in 0..dh {
                let mut sum_k = d_z[l];
                for i in 0..dh {
                    sum_k += d_kv[i * dh + l] * v_raw[h_base + i * batch];
                }
                d_k_phi[h_base + l * batch] = sum_k;
            }
            for i in 0..dh {
                let mut sum_v = 0.0f32;
                for l in 0..dh {
                    sum_v += d_kv[i * dh + l] * k_phi[h_base + l * batch];
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

        debug_assert_eq!(features, seq * d);
        debug_assert_eq!(output.rows(), batch);
        debug_assert_eq!(output.cols(), seq * d);
        debug_assert!(slice.start + self.param_len() <= params.rows() * params.cols());

        let (log_this, call_id, traj_this) = if *LINATT_DEBUG {
            let n = LINATT_FWD_CALLS.fetch_add(1, Ordering::Relaxed);
            (n < LINATT_FWD_LOG_LIMIT, n, n % LINATT_TRAJECTORY_EVERY == 0)
        } else {
            (false, 0usize, false)
        };

        let head_size = self.head_param_count();
        let h_raw_idx = slice.start + self.max_heads * head_size;

        // ------------------------------------------------------------------
        // Читаем параметры и вход, выполняем все вычисления в одном блоке.
        // x_guard и p_guard дропаются автоматически в конце этого блока,
        // а `x` и `y_accum` (для отладки) переносятся наружу как владеемые
        // векторы.
        // ------------------------------------------------------------------

        // Читаем параметры один раз, копируем нужные срезы в локальный Vec,
        // потому что нам нужны они вне блока (для отладки используются p
        // через linatt_stats, но оно читает срез, а не весь p, — здесь
        // просто хватает guard внутри блока).
        let param_vec = params.read_range(slice.start, self.param_len());

        // Читаем вход целиком один раз.
        let x_vec = input.read_range(0, total_in);

        // Инициализация и хранение пер-головых буферов.
        let mut head_buffers: Vec<CpuLinearAttentionHead> = Vec::new();
        let mut y_accum = vec![0.0f32; total_in];

        let h_raw = param_vec[h_raw_idx - slice.start];
        let h_soft = compute_h_soft(h_raw, self.min_heads, self.max_heads);

        for h in 0..self.max_heads {
            let w_h = head_weight(h_soft, h);
            if w_h <= HEAD_WEIGHT_EPS {
                continue;
            }

            let head_base = h * head_size; // внутри param_vec
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

            // Создаём буферы в пуле под промежуточные данные этой головы.
            let q_buf = pool.acquire(total_head, 1);
            let k_buf = pool.acquire(total_head, 1);
            let v_buf = pool.acquire(total_head, 1);
            let kv_buf = pool.acquire(dh * dh, 1);
            let z_buf = pool.acquire(dh, 1);
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

        // Записываем выход.
        output.write_range(0, &y_accum);

        // Отладочная диагностика.
        if *LINATT_DEBUG {
            let has_anom = linatt_has_bad(&y_accum);
            if log_this || has_anom {
                println!(
                    "[LINATT fwd #{}] batch={}, seq={}, d_model={}, d_head={}, \
                     min_heads={}, max_heads={}, h_raw={:.6}, h_soft={:.6}, \
                     active_heads={}",
                    call_id, batch, seq, d, dh,
                    self.min_heads, self.max_heads,
                    h_raw, h_soft, head_buffers.len()
                );
                for hc in &head_buffers {
                    let y_h = hc.y.read_range(0, total_in);
                    let attn = hc.attn_out.read_range(0, total_head);
                    println!(
                        "    head {}: w={:.6}, ||attn_out||={:.4}, ||y_h||={:.4}",
                        hc.head_index, hc.weight,
                        linatt_l2(&attn), linatt_l2(&y_h)
                    );
                }
                linatt_stats("x (input)", &x_vec);
                linatt_stats("y (output, weighted sum)", &y_accum);
                linatt_first("y", &y_accum, 8);
                if has_anom && !log_this {
                    println!("[LINATT fwd #{}] ANOMALY (non-finite in y)", call_id);
                }
            }
            if traj_this {
                println!(
                    "[LINATT traj fwd #{}] h_raw={:.4} h_soft={:.4} \
                     active_heads={} ||x||={:.4} ||y||={:.4}",
                    call_id, h_raw, h_soft, head_buffers.len(),
                    linatt_l2(&x_vec), linatt_l2(&y_accum)
                );
            }
        }

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

        let (log_this, call_id, traj_this) = if *LINATT_DEBUG {
            let n = LINATT_BWD_CALLS.fetch_add(1, Ordering::Relaxed);
            (n < LINATT_BWD_LOG_LIMIT, n, n % LINATT_TRAJECTORY_EVERY == 0)
        } else {
            (false, 0usize, false)
        };

        let head_size = self.head_param_count();
        let h_raw_idx = slice.start + self.max_heads * head_size;

        // ------------------------------------------------------------------
        // КРИТИЧНО (fix deadlock):
        //
        // Per-head state-буферы читаем ЗАРАНЕЕ, до входа в
        // with_cpu_slices_mut, потому что with_cpu_slices_mut держит
        // ЭКСКЛЮЗИВНУЮ блокировку MemoryExecutor, а read_range пытается
        // взять блокировку на чтение того же RwLock → deadlock (RwLock
        // не реентерабельный). Раньше этот код вызывал head_cache.*.read_range
        // внутри замыкания — и вешался на batch>1 (backward вообще не
        // вызывался при диагностическом forward=1).
        //
        // Порядок чтения совпадает с порядком в forward-е: по каждой активной
        // голове — q, k, v, kv, z, attn_out, y.
        // ------------------------------------------------------------------
        struct HeadSnapshot {
            weight: f32,
            head_index: usize,
            q_phi: Vec<f32>,
            k_phi: Vec<f32>,
            v_raw: Vec<f32>,
            kv: Vec<f32>,
            z: Vec<f32>,
            attn_out: Vec<f32>,
            y_h: Vec<f32>,
        }

        let total_head = batch * seq * dh;

        let mut snapshots: Vec<HeadSnapshot> = Vec::with_capacity(cpu_heads.len());
        for hc in cpu_heads {
            let q_phi = hc.q.read_range(0, total_head);
            let k_phi = hc.k.read_range(0, total_head);
            let v_raw = hc.v.read_range(0, total_head);
            let kv = hc.kv.read_range(0, dh * dh);
            let z = hc.z.read_range(0, dh);
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

        // Теперь безопасно берём write-блокировку один раз и работаем с
        // сырыми срезами буферов. Никаких read_range внутри замыкания.
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

                // Обнуляем градиенты параметров и входа.
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

                    // go_h = w_h * go
                    for i in 0..total_in {
                        go_h[i] = w_h * go[i];
                    }

                    // dL/dw_h = dot(go, y_h)
                    let mut d_l_dw_h = 0.0f32;
                    for i in 0..total_in {
                        d_l_dw_h += go[i] * snap.y_h[i];
                    }

                    // dL/dh_soft += dL/dw_h * dw_h/dh_soft
                    let dw_dhs = head_weight_derivative(h_soft, h);
                    d_l_dh_raw_total += d_l_dw_h * dw_dhs;

                    // Backward головы.
                    let res = compute_head_backward(
                        x, &go_h, p, head_base,
                        &snap.q_phi, &snap.k_phi, &snap.v_raw,
                        &snap.kv, &snap.z, &snap.attn_out,
                        batch, seq, d, dh,
                    );

                    for i in 0..total_in {
                        gi[i] += res.gi[i];
                    }

                    // Запись градиентов параметров головы.
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

                    if *LINATT_DEBUG && (log_this || traj_this) {
                        println!(
                            "    [LINATT bwd] head {}: w={:.6}, dL/dw={:.6e}, \
                             dL/dh_soft={:.6e}, ||gi_head||={:.4e}",
                            h, w_h, d_l_dw_h, d_l_dw_h * dw_dhs, linatt_l2(&res.gi)
                        );
                    }
                }

                let dh_soft_dh_raw = compute_h_soft_derivative(h_raw, self.min_heads, self.max_heads);
                let d_l_dh_raw = d_l_dh_raw_total * dh_soft_dh_raw;
                gp[h_raw_idx] = d_l_dh_raw;

                if *LINATT_DEBUG {
                    let has_anom = linatt_has_bad(gi)
                        || !d_l_dh_raw.is_finite()
                        || !d_l_dh_raw_total.is_finite();
                    if log_this || has_anom {
                        println!(
                            "[LINATT bwd #{}] d_model={}, d_head={}, h_raw={:.6}, \
                             h_soft={:.6}, active_heads={}, dL/dh_raw={:.6e}",
                            call_id, d, dh, h_raw, h_soft,
                            snapshots.len(), d_l_dh_raw
                        );
                        linatt_stats("go (grad_out)", go);
                        linatt_stats("gi (grad_input)", gi);
                        if has_anom && !log_this {
                            println!("[LINATT bwd #{}] ANOMALY", call_id);
                        }
                    }
                    if traj_this {
                        println!(
                            "[LINATT traj bwd #{}] h_raw={:.4} h_soft={:.4} \
                             active_heads={} dL/dh_raw={:.4e} ||go||={:.4e} ||gi||={:.4e}",
                            call_id, h_raw, h_soft, snapshots.len(),
                            d_l_dh_raw, linatt_l2(go), linatt_l2(gi)
                        );
                    }
                }
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