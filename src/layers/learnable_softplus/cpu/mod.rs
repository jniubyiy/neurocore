// src/layers/learnable_softplus/cpu/mod.rs

use std::sync::atomic::{AtomicUsize, Ordering};

use once_cell::sync::Lazy;

use crate::compute_manager::graph::types::DynamicContext;
use crate::compute_manager::matrix_buffer::MatrixBufferHandle;
use crate::layers::buffered_context::BufferedContext;
use crate::layers::UniversalLayerBuffered;
use crate::model_plan::param_store::ParamSlice;

use super::super::learnable_softplus::LearnableSoftplus;

// ============================================================================
// Параметризация β и математическое обоснование
// ============================================================================
//
//   β = exp(log_beta)
//
// Это ЕДИНСТВЕННАЯ гладкая параметризация, устраняющая сингулярность 1/β
// в ∂y/∂β:
//
//   y = softplus(β·u)/β,   u = x − θ
//   ∂y/∂β = (σ(β·u)·u − y)/β
//   ∂y/∂log_beta = ∂y/∂β · β = σ(β·u)·u − y    ← без 1/β
//
// Любая другая f (softplus, sigmoid, ...) оставляет 1/β в градиенте.
// ============================================================================

static LSF_DEBUG: Lazy<bool> = Lazy::new(|| std::env::var("NEUROCORE_DEBUG_LSF").is_ok());
static LSF_FWD_CALLS: Lazy<AtomicUsize> = Lazy::new(|| AtomicUsize::new(0));
static LSF_BWD_CALLS: Lazy<AtomicUsize> = Lazy::new(|| AtomicUsize::new(0));

// Подробные логи первых N forward/backward (значения, распределения).
const FORWARD_LOG_LIMIT: usize = 3;
const BACKWARD_LOG_LIMIT: usize = 3;

// Траектория β: печатается каждые LOG_TRAJECTORY_EVERY forward-вызовов.
const LOG_TRAJECTORY_EVERY: usize = 50;

// Диагностика backward при попадании в траекторию.
// (нужно понять, куда β движется на всём протяжении обучения)
const LOG_BWD_TRAJECTORY_EVERY: usize = 50;

// ----------------------------------------------------------------------------
// Границы β
// ----------------------------------------------------------------------------
//
// Нижний clamp нужен: при β → 0 формула y = softplus(β·u)/β → ln(2)/β → +∞.
//
// Верхний clamp УБРАН: чтобы восстановить x = 0.01, нужна β ≈ 100.
// β ≈ 1e8 безопасна (y → u, identity). До inf дойдёт только через decades
// экспоненты — практически невозможно.
// ----------------------------------------------------------------------------

const BETA_MIN: f32 = 1e-3;
const LOG_BETA_MIN: f32 = -6.907755;  // ln(1e-3)

/// Печатает статистику по срезу: min, max, mean, кол-во NaN/Inf.
fn lsf_dbg_stats(name: &str, data: &[f32]) {
    if data.is_empty() {
        println!("    [LSF] {}: <empty>", name);
        return;
    }
    let mut mn = f32::INFINITY;
    let mut mx = f32::NEG_INFINITY;
    let mut sum = 0.0f64;
    let mut nan_cnt = 0usize;
    let mut inf_cnt = 0usize;
    for &v in data {
        if v.is_nan() {
            nan_cnt += 1;
            continue;
        }
        if v.is_infinite() {
            inf_cnt += 1;
            continue;
        }
        if v < mn { mn = v; }
        if v > mx { mx = v; }
        sum += v as f64;
    }
    let finite = data.len().saturating_sub(nan_cnt + inf_cnt);
    let mean = if finite > 0 { sum / finite as f64 } else { 0.0 };
    println!(
        "    [LSF] {}: len={}, min={:.6}, max={:.6}, mean={:.6}, nan={}, inf={}",
        name, data.len(), mn, mx, mean, nan_cnt, inf_cnt
    );
}

/// Устойчивый softplus: log(1 + exp(z)) без переполнения.
#[inline]
fn softplus_stable(z: f32) -> f32 {
    if z > 0.0 {
        z + (-z).exp().ln_1p()
    } else {
        z.exp().ln_1p()
    }
}

/// Применяет clamp к log_beta (только снизу) и возвращает
/// (log_beta_eff, beta_eff, is_clamped).
#[inline]
fn effective_beta(log_beta: f32) -> (f32, f32, bool) {
    if log_beta < LOG_BETA_MIN {
        (LOG_BETA_MIN, BETA_MIN, true)
    } else {
        (log_beta, log_beta.exp(), false)
    }
}

/// Краткая сводка β для траекторного лога.
fn lsf_traj_print(label: &str, log_betas: &[f32], y: &[f32]) {
    let (betas_eff, n_clamped) = {
        let mut betas = Vec::with_capacity(log_betas.len());
        let mut n_clamped = 0usize;
        for &lb in log_betas {
            let (_, b, c) = effective_beta(lb);
            betas.push(b);
            if c { n_clamped += 1; }
        }
        (betas, n_clamped)
    };
    let b_min = betas_eff.iter().cloned().fold(f32::INFINITY, f32::min);
    let b_max = betas_eff.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let b_mean = betas_eff.iter().sum::<f32>() / betas_eff.len() as f32;
    let y_min = y.iter().cloned().fold(f32::INFINITY, f32::min);
    let y_max = y.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    println!(
        "[LSF traj {}] beta: min={:.4} max={:.4} mean={:.4}, clamped={}/{} | y: min={:.4} max={:.4}",
        label, b_min, b_max, b_mean, n_clamped, log_betas.len(), y_min, y_max
    );
}

impl UniversalLayerBuffered for LearnableSoftplus {
    fn forward_buffered(
        &self,
        input: &MatrixBufferHandle,
        output: &MatrixBufferHandle,
        params: &MatrixBufferHandle,
        slice: &ParamSlice,
    ) {
        let rows = input.rows();
        let cols = input.cols();
        debug_assert_eq!(cols, self.features);
        debug_assert!(
            slice.start + self.param_len() <= params.rows() * params.cols(),
            "LearnableSoftplus: parameter slice out of bounds"
        );

        let (lsf_log_this, lsf_call_id, lsf_traj_this) = if *LSF_DEBUG {
            let n = LSF_FWD_CALLS.fetch_add(1, Ordering::Relaxed);
            let log_this = n < FORWARD_LOG_LIMIT;
            let traj_this = n % LOG_TRAJECTORY_EVERY == 0;
            (log_this, n, traj_this)
        } else {
            (false, 0usize, false)
        };

        let ids = [input.id(), output.id(), params.id()];
        input.memory().write().unwrap().with_cpu_slices_mut(&ids, |slices| {
            let (first, rest) = slices.split_at_mut(1);
            let x: &[f32] = &*first[0];
            let (second, rest) = rest.split_at_mut(1);
            let y: &mut [f32] = &mut *second[0];
            let p: &[f32] = &*rest[0];

            let log_beta_start = slice.start;
            let theta_start = log_beta_start + self.features;

            if lsf_log_this {
                println!(
                    "[LSF fwd #{}] rows={}, cols={}, slice.start={}, features={}",
                    lsf_call_id, rows, cols, slice.start, self.features
                );
                let log_betas = &p[log_beta_start..log_beta_start + self.features];
                let thetas = &p[theta_start..theta_start + self.features];
                lsf_dbg_stats("log_beta (raw)", log_betas);
                lsf_dbg_stats("theta (raw)", thetas);
                let betas_eff: Vec<f32> = log_betas
                    .iter()
                    .map(|&lb| effective_beta(lb).1)
                    .collect();
                lsf_dbg_stats("beta_eff = exp(log_beta)", &betas_eff);
                lsf_dbg_stats("x (input)", x);
                let show = log_betas.len().min(8);
                println!("    first log_betas: {:?}", &log_betas[..show]);
                println!("    first betas_eff: {:?}", &betas_eff[..show]);
                println!("    first thetas:    {:?}", &thetas[..show]);
            }

            for c in 0..cols {
                let log_beta = p[log_beta_start + c];
                let (_, beta, _) = effective_beta(log_beta);
                let inv_beta = 1.0 / beta;
                let theta = p[theta_start + c];

                for r in 0..rows {
                    let idx = c * rows + r;
                    let x_val = x[idx];
                    let shifted = beta * (x_val - theta);
                    y[idx] = inv_beta * softplus_stable(shifted);
                }
            }

            if *LSF_DEBUG {
                let has_anom = y.iter().any(|v| !v.is_finite());
                if lsf_log_this || has_anom {
                    if has_anom && !lsf_log_this {
                        println!("[LSF fwd #{}] ANOMALY (non-finite in output)", lsf_call_id);
                    }
                    lsf_dbg_stats("y (output)", y);
                }
                // Траектория β
                if lsf_traj_this {
                    let log_betas = &p[log_beta_start..log_beta_start + self.features];
                    lsf_traj_print(&format!("#{}", lsf_call_id), log_betas, y);
                }
            }
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
            BufferedContext::LearnableSoftplus { input } => input,
            _ => panic!("Expected LearnableSoftplus context"),
        };

        let rows = grad_output.rows();
        let cols = grad_output.cols();
        debug_assert_eq!(cols, self.features);
        debug_assert_eq!(rows, input_handle.rows());
        debug_assert!(
            slice.start + self.param_len() <= params.rows() * params.cols(),
            "LearnableSoftplus backward: parameter slice out of bounds"
        );
        debug_assert!(
            slice.start + self.param_len() <= grad_params.rows() * grad_params.cols(),
            "LearnableSoftplus backward: grad parameter slice out of bounds"
        );

        let (lsf_log_this, lsf_call_id, lsf_traj_this) = if *LSF_DEBUG {
            let n = LSF_BWD_CALLS.fetch_add(1, Ordering::Relaxed);
            let log_this = n < BACKWARD_LOG_LIMIT;
            let traj_this = n % LOG_BWD_TRAJECTORY_EVERY == 0;
            (log_this, n, traj_this)
        } else {
            (false, 0usize, false)
        };

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

                let log_beta_start = slice.start;
                let theta_start = log_beta_start + self.features;

                if lsf_log_this {
                    println!(
                        "[LSF bwd #{}] rows={}, cols={}, slice.start={}",
                        lsf_call_id, rows, cols, slice.start
                    );
                    let log_betas = &p[log_beta_start..log_beta_start + self.features];
                    lsf_dbg_stats("log_beta (raw)", log_betas);
                    let betas_eff: Vec<f32> = log_betas
                        .iter()
                        .map(|&lb| effective_beta(lb).1)
                        .collect();
                    lsf_dbg_stats("beta_eff", &betas_eff);
                    lsf_dbg_stats("x (input)", x);
                    lsf_dbg_stats("go (grad_out)", go);
                }

                let mut grad_log_beta = vec![0.0f32; self.features];
                let mut grad_theta = vec![0.0f32; self.features];

                for c in 0..cols {
                    let log_beta = p[log_beta_start + c];
                    let (_, beta, is_clamped) = effective_beta(log_beta);
                    let inv_beta = 1.0 / beta;
                    let theta = p[theta_start + c];

                    let mut d_log_beta_acc = 0.0f32;
                    let mut d_theta_acc = 0.0f32;

                    for r in 0..rows {
                        let idx = c * rows + r;
                        let x_val = x[idx];
                        let gout = go[idx];
                        let u = x_val - theta;

                        let shifted = beta * u;
                        let sigmoid = 1.0 / (1.0 + (-shifted).exp());
                        let y_val = inv_beta * softplus_stable(shifted);

                        // ∂y/∂x = σ(β·u)
                        gi[idx] = gout * sigmoid;

                        // ∂y/∂log_beta = σ(β·u)·u − y
                        // (без множителя β — это результат chain rule через
                        //  f(β_raw)=exp(β_raw), f'=β, устраняющий 1/β).
                        // Если β зажат снизу — градиент через него не течёт.
                        if !is_clamped {
                            let d_log_beta = sigmoid * u - y_val;
                            d_log_beta_acc += gout * d_log_beta;
                        }

                        // ∂y/∂θ = −σ(β·u)
                        let d_theta = -sigmoid;
                        d_theta_acc += gout * d_theta;
                    }

                    grad_log_beta[c] = d_log_beta_acc;
                    grad_theta[c] = d_theta_acc;
                }

                if *LSF_DEBUG {
                    let has_anom = grad_log_beta.iter().any(|v| !v.is_finite())
                        || grad_theta.iter().any(|v| !v.is_finite())
                        || gi.iter().any(|v| !v.is_finite());
                    if lsf_log_this || has_anom {
                        if has_anom && !lsf_log_this {
                            println!("[LSF bwd #{}] ANOMALY (non-finite grad)", lsf_call_id);
                        }
                        lsf_dbg_stats("grad_log_beta", &grad_log_beta);
                        lsf_dbg_stats("grad_theta", &grad_theta);
                        lsf_dbg_stats("gi (grad_input)", gi);
                    }
                    // Траектория: краткая сводка градиента по log_beta и θ.
                    if lsf_traj_this {
                        let gl_min = grad_log_beta.iter().cloned().fold(f32::INFINITY, f32::min);
                        let gl_max = grad_log_beta.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                        let gl_mean = grad_log_beta.iter().sum::<f32>() / grad_log_beta.len() as f32;
                        let gt_min = grad_theta.iter().cloned().fold(f32::INFINITY, f32::min);
                        let gt_max = grad_theta.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                        let gt_mean = grad_theta.iter().sum::<f32>() / grad_theta.len() as f32;
                        println!(
                            "[LSF bwd-traj #{}] grad_log_beta: min={:.4} max={:.4} mean={:.4} | grad_theta: min={:.4} max={:.4} mean={:.4}",
                            lsf_call_id, gl_min, gl_max, gl_mean, gt_min, gt_max, gt_mean
                        );
                    }
                }

                for c in 0..self.features {
                    gp[log_beta_start + c] = grad_log_beta[c];
                    gp[theta_start + c] = grad_theta[c];
                }
            });
    }

    fn param_len(&self) -> usize {
        2 * self.features
    }

    fn input_features(&self) -> usize {
        self.features
    }

    fn output_features(&self) -> usize {
        self.features
    }
}