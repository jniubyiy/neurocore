// src/layers/batch_renorm/cpu/mod.rs

use std::sync::atomic::{AtomicUsize, Ordering};

use once_cell::sync::Lazy;

use crate::compute_manager::graph::types::DynamicContext;
use crate::compute_manager::matrix_buffer::{MatrixBufferHandle, TempMatrixPool};
use crate::layers::buffered_context::BufferedContext;
use crate::layers::UniversalLayerBuffered;
use crate::model_plan::param_store::ParamSlice;

use super::super::batch_renorm::BatchRenorm1d;

// ============================================================================
// Диагностика BatchRenorm1d (CPU)
// ============================================================================
//
// Включается переменной окружения NEUROCORE_DEBUG_BATCHRENORM=1.
//
// Логируются:
//   * первые BR_FWD_LOG_LIMIT вызовов forward — статистика по параметрам,
//     входу, batch-статистикам и выходу;
//   * каждый BR_TRAJ_EVERY-й forward — компактная траектория;
//   * первые BR_BWD_LOG_LIMIT вызовов backward — статистика go, gi,
//     градиентов по параметрам и L2-нормы term1/term2/term3;
//   * каждый BR_BWD_TRAJ_EVERY-й backward — компактная траектория.

static BR_DEBUG: Lazy<bool> =
    Lazy::new(|| std::env::var("NEUROCORE_DEBUG_BATCHRENORM").is_ok());

static BR_FWD_CALLS: Lazy<AtomicUsize> = Lazy::new(|| AtomicUsize::new(0));
static BR_BWD_CALLS: Lazy<AtomicUsize> = Lazy::new(|| AtomicUsize::new(0));

const BR_FWD_LOG_LIMIT: usize = 3;
const BR_BWD_LOG_LIMIT: usize = 3;
const BR_TRAJ_EVERY: usize = 50;
const BR_BWD_TRAJ_EVERY: usize = 50;

fn br_stats(name: &str, data: &[f32]) {
    if data.is_empty() {
        println!("    [BR] {}: <empty>", name);
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
    let l2: f64 = data
        .iter()
        .filter(|v| v.is_finite())
        .map(|&v| (v as f64) * (v as f64))
        .sum::<f64>()
        .sqrt();
    println!(
        "    [BR] {}: len={}, min={:.6}, max={:.6}, mean={:.6}, l2={:.6}, nan={}, inf={}",
        name, data.len(), mn, mx, mean, l2, nan_cnt, inf_cnt
    );
}

fn br_first(label: &str, data: &[f32], k: usize) {
    let show = data.len().min(k);
    println!("    [BR] {} (first {}): {:?}", label, show, &data[..show]);
}

fn br_has_bad(data: &[f32]) -> bool {
    data.iter().any(|v| !v.is_finite())
}

impl UniversalLayerBuffered for BatchRenorm1d {
    fn forward_buffered(
        &self,
        input: &MatrixBufferHandle,
        output: &MatrixBufferHandle,
        params: &MatrixBufferHandle,
        slice: &ParamSlice,
        _pool: &mut TempMatrixPool,
    ) -> BufferedContext {
        let rows = input.rows();
        let cols = input.cols();
        debug_assert_eq!(cols, self.features);
        debug_assert!(
            slice.start + self.param_len() <= params.rows() * params.cols(),
            "BatchRenorm1d: parameter slice out of bounds"
        );

        let f = self.features;
        let eps = self.eps;
        let momentum = self.momentum;

        // Локальные копии статистик и режима — забираем под одним read().
        let (training, running_mean_local, running_var_local) = {
            let state = self.state.read().unwrap();
            (
                state.training,
                state.running_mean.clone(),
                state.running_var.clone(),
            )
        };

        let (log_this, call_id, traj_this) = if *BR_DEBUG {
            let n = BR_FWD_CALLS.fetch_add(1, Ordering::Relaxed);
            (n < BR_FWD_LOG_LIMIT, n, n % BR_TRAJ_EVERY == 0)
        } else {
            (false, 0usize, false)
        };

        let ids = [input.id(), output.id(), params.id()];

        let (mean, var) = input
            .memory()
            .write()
            .unwrap()
            .with_cpu_slices_mut(&ids, |slices| {
                let (first, rest) = slices.split_at_mut(1);
                let x: &[f32] = &*first[0];
                let (second, rest) = rest.split_at_mut(1);
                let y: &mut [f32] = &mut *second[0];
                let (third, _) = rest.split_at_mut(1);
                let p: &[f32] = &*third[0];

                let base = slice.start;
                let gamma_start = base;
                let beta_start = gamma_start + f;
                let r_start = beta_start + f;
                let d_start = r_start + f;

                // ============ 1. Статистики текущего прохода. ============
                let (mean, var): (Vec<f32>, Vec<f32>) = if training {
                    let mut batch_mean = vec![0.0f32; f];
                    let mut batch_var = vec![0.0f32; f];
                    for c in 0..cols {
                        let mut sum = 0.0f32;
                        let mut sum_sq = 0.0f32;
                        for r in 0..rows {
                            let idx = c * rows + r;
                            let v = x[idx];
                            sum += v;
                            sum_sq += v * v;
                        }
                        let mean_c = sum / rows as f32;
                        let var_c = (sum_sq / rows as f32) - mean_c * mean_c;
                        batch_mean[c] = mean_c;
                        batch_var[c] = var_c.max(0.0f32);
                    }
                    (batch_mean, batch_var)
                } else {
                    (running_mean_local.clone(), running_var_local.clone())
                };

                // ============ ДИАГНОСТИКА: forward ============
                if *BR_DEBUG && log_this {
                    println!(
                        "[BR fwd #{}] rows={}, cols={}, slice.start={}, features={}, training={}",
                        call_id, rows, cols, slice.start, f, training
                    );
                    let gammas = &p[gamma_start..gamma_start + f];
                    let betas  = &p[beta_start..beta_start + f];
                    let r_pars = &p[r_start..r_start + f];
                    let d_pars = &p[d_start..d_start + f];
                    br_stats("gamma", gammas);
                    br_stats("beta",  betas);
                    br_stats("r",     r_pars);
                    br_stats("d",     d_pars);
                    br_stats("x (input)", x);
                    br_stats("mean (per feature)", &mean);
                    br_stats("var  (per feature)", &var);
                    br_stats("running_mean (state)", &running_mean_local);
                    br_stats("running_var  (state)", &running_var_local);
                    br_first("gamma", gammas, 8);
                    br_first("beta",  betas, 8);
                    br_first("r",     r_pars, 8);
                    br_first("d",     d_pars, 8);
                    br_first("mean",  &mean, 8);
                    br_first("var",   &var, 8);
                } else if *BR_DEBUG && br_has_bad(x) {
                    println!("[BR fwd #{}] ANOMALY: non-finite in input x", call_id);
                    br_stats("x (input)", x);
                }

                // ============ 2. Прямой проход. ============
                for c in 0..cols {
                    let gamma = p[gamma_start + c];
                    let beta = p[beta_start + c];
                    let r_par = p[r_start + c];
                    let d_par = p[d_start + c];
                    let mean_c = mean[c];
                    let var_c = var[c];
                    let inv_std = 1.0 / (var_c + eps).sqrt();
                    for row in 0..rows {
                        let idx = c * rows + row;
                        let x_hat = (x[idx] - mean_c) * inv_std;
                        y[idx] = x_hat * r_par * gamma + d_par * gamma + beta;
                    }
                }

                // ============ ДИАГНОСТИКА: выход ============
                if *BR_DEBUG && log_this {
                    br_stats("y (output)", y);
                    br_first("y", y, 8);
                } else if *BR_DEBUG && br_has_bad(y) {
                    println!("[BR fwd #{}] ANOMALY: non-finite in output y", call_id);
                    br_stats("y (output)", y);
                }

                if *BR_DEBUG && traj_this {
                    let gammas = &p[gamma_start..gamma_start + f];
                    let r_pars = &p[r_start..r_start + f];
                    let betas  = &p[beta_start..beta_start + f];
                    let d_pars = &p[d_start..d_start + f];
                    let g_min = gammas.iter().cloned().fold(f32::INFINITY, f32::min);
                    let g_max = gammas.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let r_min = r_pars.iter().cloned().fold(f32::INFINITY, f32::min);
                    let r_max = r_pars.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let b_abs = betas.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                    let d_abs = d_pars.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                    let m_mean = mean.iter().sum::<f32>() / f as f32;
                    let v_mean = var.iter().sum::<f32>()  / f as f32;
                    println!(
                        "[BR traj fwd #{}] training={} | γ∈[{:.4},{:.4}] r∈[{:.4},{:.4}] |β|∞={:.4} |d|∞={:.4} | mean̄={:.4} var̄={:.4}",
                        call_id, training, g_min, g_max, r_min, r_max, b_abs, d_abs, m_mean, v_mean
                    );
                }

                (mean, var)
            });

        // ============ 3. Обновление running-статистик (если обучаемся). ============
        if training {
            let mut state = self.state.write().unwrap();
            for c in 0..f {
                state.running_mean[c] = (1.0 - momentum) * state.running_mean[c]
                    + momentum * mean[c];
                state.running_var[c] = (1.0 - momentum) * state.running_var[c]
                    + momentum * var[c];
            }
        }

        BufferedContext::BatchRenorm {
            input: input.clone(),
            mean,
            var,
            use_batch_stats: training,
        }
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
        let (input_handle, mean, var, use_batch_stats) = match bc {
            BufferedContext::BatchRenorm {
                input,
                mean,
                var,
                use_batch_stats,
            } => (input, mean, var, *use_batch_stats),
            _ => panic!("Expected BatchRenorm Buffered context"),
        };

        let rows = grad_output.rows();
        let cols = grad_output.cols();
        debug_assert_eq!(cols, self.features);
        debug_assert_eq!(rows, input_handle.rows());
        debug_assert_eq!(mean.len(), self.features);
        debug_assert_eq!(var.len(), self.features);
        debug_assert!(
            slice.start + self.param_len() <= params.rows() * params.cols(),
            "BatchRenorm1d backward: parameter slice out of bounds"
        );
        debug_assert!(
            slice.start + self.param_len() <= grad_params.rows() * grad_params.cols(),
            "BatchRenorm1d backward: grad parameter slice out of bounds"
        );

        let f = self.features;
        let eps = self.eps;

        let (log_this, call_id, traj_this) = if *BR_DEBUG {
            let n = BR_BWD_CALLS.fetch_add(1, Ordering::Relaxed);
            (n < BR_BWD_LOG_LIMIT, n, n % BR_BWD_TRAJ_EVERY == 0)
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
                let (fifth, _) = rest.split_at_mut(1);
                let gp: &mut [f32] = &mut *fifth[0];

                let base = slice.start;
                let gamma_start = base;
                let beta_start = gamma_start + f;
                let r_start = beta_start + f;
                let d_start = r_start + f;

                // Обнуляем градиенты параметров.
                for i in 0..(4 * f) {
                    gp[base + i] = 0.0f32;
                }

                let mut grad_gamma = vec![0.0f32; f];
                let mut grad_beta = vec![0.0f32; f];
                let mut grad_r = vec![0.0f32; f];
                let mut grad_d = vec![0.0f32; f];

                let mut diag_term1_l2: f64 = 0.0;
                let mut diag_term2_l2: f64 = 0.0;
                let mut diag_term3_l2: f64 = 0.0;

                // ============ 1. Градиенты по параметрам. ============
                for c in 0..cols {
                    let gamma = p[gamma_start + c];
                    let r_par = p[r_start + c];
                    let d_par = p[d_start + c];
                    let mean_c = mean[c];
                    let var_c = var[c];
                    let inv_std = 1.0 / (var_c + eps).sqrt();

                    let mut sum_gamma_r = 0.0f32;
                    let mut sum_gamma_r_xhat = 0.0f32;

                    for row in 0..rows {
                        let idx = c * rows + row;
                        let x_val = x[idx];
                        let gout = go[idx];
                        let x_hat = (x_val - mean_c) * inv_std;

                        grad_gamma[c] += gout * (x_hat * r_par + d_par);
                        grad_beta[c] += gout;
                        grad_r[c] += gout * gamma * x_hat;
                        grad_d[c] += gout * gamma;

                        if use_batch_stats {
                            let gy = gout * gamma * r_par;
                            sum_gamma_r += gy;
                            sum_gamma_r_xhat += gy * x_hat;
                        }
                    }

                    // ============ 2. Градиент по входу. ============
                    //
                    // Каноническая формула BN-backward для y = γ_eff · x_hat + β_eff:
                    //
                    //   gi_i = (γ_eff/σ) · [ g_i − mean(g) − x_hat_i · mean(g·x_hat) ]
                    //
                    // где γ_eff = γ·r. Раскрывая mean() через Σ/n, получаем:
                    //
                    //   term1 = g_i · γ · r / σ
                    //   term2 = Σ_j g_j · γ · r / (n · σ)      ← деление на σ обязательно
                    //   term3 = x_hat_i · Σ_j g_j · γ · r · x_hat_j / (n · σ)
                    //
                    // FIX: ранее term2 считался без деления на σ, что давало
                    // неверный градиент по входу при σ ≠ 1.
                    let sigma = (var_c + eps).sqrt();
                    for row in 0..rows {
                        let idx = c * rows + row;
                        let gout = go[idx];
                        let x_hat = (x[idx] - mean_c) * inv_std;

                        if use_batch_stats {
                            let n = rows as f32;
                            let term1 = gout * gamma * r_par * inv_std;
                            let term2 = sum_gamma_r / (n * sigma);
                            let term3 = x_hat * sum_gamma_r_xhat / (n * sigma);
                            gi[idx] = term1 - term2 - term3;

                            if *BR_DEBUG && log_this {
                                diag_term1_l2 += (term1 as f64) * (term1 as f64);
                                diag_term2_l2 += (term2 as f64) * (term2 as f64);
                                diag_term3_l2 += (term3 as f64) * (term3 as f64);
                            }
                        } else {
                            gi[idx] = gout * gamma * r_par * inv_std;
                        }
                    }
                }

                // ============ 3. Запись градиентов параметров. ============
                for c in 0..f {
                    gp[gamma_start + c] = grad_gamma[c];
                    gp[beta_start + c] = grad_beta[c];
                    gp[r_start + c] = grad_r[c];
                    gp[d_start + c] = grad_d[c];
                }

                // ============ ДИАГНОСТИКА ============
                if *BR_DEBUG && log_this {
                    println!(
                        "[BR bwd #{}] rows={}, cols={}, slice.start={}, use_batch_stats={}",
                        call_id, rows, cols, slice.start, use_batch_stats
                    );
                    br_stats("go (grad_out)", go);
                    br_stats("gi (grad_input)", gi);
                    br_stats("x (input)", x);
                    br_stats("grad_gamma", &grad_gamma);
                    br_stats("grad_beta",  &grad_beta);
                    br_stats("grad_r",     &grad_r);
                    br_stats("grad_d",     &grad_d);
                    br_first("grad_gamma", &grad_gamma, 8);
                    br_first("grad_beta",  &grad_beta, 8);
                    br_first("grad_r",     &grad_r, 8);
                    br_first("grad_d",     &grad_d, 8);

                    if use_batch_stats {
                        let t1 = diag_term1_l2.sqrt();
                        let t2 = diag_term2_l2.sqrt();
                        let t3 = diag_term3_l2.sqrt();
                        println!(
                            "    [BR] term1_l2={:.6}, term2_l2={:.6}, term3_l2={:.6}",
                            t1, t2, t3
                        );
                    }
                } else if *BR_DEBUG {
                    let bad = br_has_bad(gi)
                        || br_has_bad(&grad_gamma)
                        || br_has_bad(&grad_beta)
                        || br_has_bad(&grad_r)
                        || br_has_bad(&grad_d);
                    if bad {
                        println!("[BR bwd #{}] ANOMALY: non-finite grad", call_id);
                        br_stats("go", go);
                        br_stats("gi", gi);
                        br_stats("grad_gamma", &grad_gamma);
                        br_stats("grad_beta",  &grad_beta);
                        br_stats("grad_r",     &grad_r);
                        br_stats("grad_d",     &grad_d);
                    }
                }

                if *BR_DEBUG && traj_this {
                    let gg_l2: f64 = grad_gamma
                        .iter().filter(|v| v.is_finite())
                        .map(|&v| (v as f64) * (v as f64)).sum::<f64>().sqrt();
                    let gb_l2: f64 = grad_beta
                        .iter().filter(|v| v.is_finite())
                        .map(|&v| (v as f64) * (v as f64)).sum::<f64>().sqrt();
                    let gr_l2: f64 = grad_r
                        .iter().filter(|v| v.is_finite())
                        .map(|&v| (v as f64) * (v as f64)).sum::<f64>().sqrt();
                    let gd_l2: f64 = grad_d
                        .iter().filter(|v| v.is_finite())
                        .map(|&v| (v as f64) * (v as f64)).sum::<f64>().sqrt();
                    let gi_l2: f64 = gi
                        .iter().filter(|v| v.is_finite())
                        .map(|&v| (v as f64) * (v as f64)).sum::<f64>().sqrt();
                    let go_l2: f64 = go
                        .iter().filter(|v| v.is_finite())
                        .map(|&v| (v as f64) * (v as f64)).sum::<f64>().sqrt();
                    println!(
                        "[BR traj bwd #{}] ||go||={:.6e} ||gi||={:.6e} ||grad_γ||={:.6e} ||grad_β||={:.6e} ||grad_r||={:.6e} ||grad_d||={:.6e}",
                        call_id, go_l2, gi_l2, gg_l2, gb_l2, gr_l2, gd_l2
                    );
                }
            });
    }

    fn param_len(&self) -> usize {
        4 * self.features
    }

    fn input_features(&self) -> usize {
        self.features
    }

    fn output_features(&self) -> usize {
        self.features
    }
}