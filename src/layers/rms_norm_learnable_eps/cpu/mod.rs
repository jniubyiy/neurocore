// src/layers/rms_norm_learnable_eps/cpu/mod.rs

use std::sync::atomic::{AtomicUsize, Ordering};

use once_cell::sync::Lazy;

use crate::compute_manager::graph::types::DynamicContext;
use crate::compute_manager::matrix_buffer::MatrixBufferHandle;
use crate::layers::buffered_context::BufferedContext;
use crate::layers::UniversalLayerBuffered;
use crate::model_plan::param_store::ParamSlice;

use super::super::rms_norm_learnable_eps::{RMSNormWithLearnableEpsilon, EPS_MIN};

// ============================================================================
// Отладочные переключатели RMSNormWithLearnableEpsilon (CPU)
// ============================================================================
//
// Логи включаются переменной окружения NEUROCORE_DEBUG_RMSNORM=1.
//
// Что печатается:
//   * первые FORWARD_LOG_LIMIT forward-вызовов: gamma, eps_raw, eps_eff
//     (= EPS_MIN + exp(eps_raw)), x, mean_sq, y, плюс диагностика
//     min(mean_sq + eps_eff) и счётчик «плохих» элементов
//     (mean_sq + eps_eff <= 0);
//   * первые BACKWARD_LOG_LIMIT backward-вызовов: grad_gamma, grad_eps_raw,
//     gi (grad_input);
//   * ЛЮБЫЕ вызовы с NaN/Inf в выходе/градиентах, даже после исчерпания
//     лимита подробных логов;
//   * траектория параметров и градиентов каждые LOG_TRAJECTORY_EVERY
//     вызовов forward и backward соответственно.
//
// Обеспечение корректности формулы ∂L/∂x (фикс B):
//   Реализованный backward вычисляет
//     gi_c = go_c·γ_c/d_c  −  (x_c / F) · Σ_j  go_j·γ_j·x_j / d_j³,
//   где d_j = sqrt(mean_sq + ε_j), F = features, ε_j — per-feature обучаемый
//   эффективный эпсилон. Это аналитически точная производная от forward-
//   формулы y_c = γ_c·x_c/d_c при per-feature ε_c (улучшение слоя
//   относительно PyTorch-канона с общим скалярным ε сохранено).
//
//   В пределе ε_j = const (обычный PyTorch RMSNorm) формула сводится к
//   PyTorch-канонической; при разных ε_j учитывает per-feature природу
//   каждого знаменателя под суммой (d_j³ внутри суммы, а не d_c³ снаружи).

static RMSN_DEBUG: Lazy<bool> =
    Lazy::new(|| std::env::var("NEUROCORE_DEBUG_RMSNORM").is_ok());
static RMSN_FWD_CALLS: Lazy<AtomicUsize> = Lazy::new(|| AtomicUsize::new(0));
static RMSN_BWD_CALLS: Lazy<AtomicUsize> = Lazy::new(|| AtomicUsize::new(0));

const FORWARD_LOG_LIMIT: usize = 3;
const BACKWARD_LOG_LIMIT: usize = 3;
const LOG_TRAJECTORY_EVERY: usize = 50;
const LOG_BWD_TRAJECTORY_EVERY: usize = 50;

// ----------------------------------------------------------------------------
// Утилиты диагностики
// ----------------------------------------------------------------------------

fn rmsn_stats(name: &str, data: &[f32]) {
    if data.is_empty() {
        println!("    [RMSN] {}: <empty>", name);
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
    let l2: f64 = data
        .iter()
        .filter(|v| v.is_finite())
        .map(|&v| (v as f64) * (v as f64))
        .sum::<f64>()
        .sqrt();
    println!(
        "    [RMSN] {}: len={}, min={:.6}, max={:.6}, mean={:.6}, l2={:.6}, nan={}, inf={}",
        name, data.len(), mn, mx, mean, l2, nan_cnt, inf_cnt
    );
}

fn rmsn_first(label: &str, data: &[f32], k: usize) {
    let show = data.len().min(k);
    println!("    [RMSN] {} (first {}): {:?}", label, show, &data[..show]);
}

fn rmsn_has_bad(data: &[f32]) -> bool {
    data.iter().any(|v| !v.is_finite())
}

/// Вычисляет `ε = EPS_MIN + exp(eps_raw)` поэлементно.
fn compute_eps_eff(eps_raw: &[f32]) -> Vec<f32> {
    eps_raw.iter().map(|&r| EPS_MIN + r.exp()).collect()
}

// ----------------------------------------------------------------------------
// Forward
// ----------------------------------------------------------------------------

impl UniversalLayerBuffered for RMSNormWithLearnableEpsilon {
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
            "RMSNormWithLearnableEpsilon: parameter slice out of bounds"
        );

        let (log_this, call_id, traj_this) = if *RMSN_DEBUG {
            let n = RMSN_FWD_CALLS.fetch_add(1, Ordering::Relaxed);
            (n < FORWARD_LOG_LIMIT, n, n % LOG_TRAJECTORY_EVERY == 0)
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

            let gamma_start = slice.start;
            let eps_start = gamma_start + self.features;

            // mean(x^2) для каждой строки.
            let mut mean_sq = vec![0.0f32; rows];
            for r in 0..rows {
                let mut sum_sq = 0.0f32;
                for c in 0..cols {
                    let idx = c * rows + r;
                    let v = x[idx];
                    sum_sq += v * v;
                }
                mean_sq[r] = sum_sq / cols as f32;
            }

            // Параметризация: ε = EPS_MIN + exp(eps_raw).
            // ε гарантированно > 0, поэтому mean_sq + ε всегда > 0.
            let eps_raw = &p[eps_start..eps_start + self.features];
            let eps_eff = compute_eps_eff(eps_raw);

            // Диагностика.
            let mut bad_arg_count = 0usize;
            let mut min_arg = f32::INFINITY;
            for c in 0..cols {
                let eps = eps_eff[c];
                for r in 0..rows {
                    let arg = mean_sq[r] + eps;
                    if arg < min_arg {
                        min_arg = arg;
                    }
                    if arg <= 0.0 {
                        bad_arg_count += 1;
                    }
                }
            }

            if *RMSN_DEBUG && log_this {
                println!(
                    "[RMSN fwd #{}] rows={}, cols={}, slice.start={}, features={}",
                    call_id, rows, cols, slice.start, self.features
                );
                let gammas = &p[gamma_start..gamma_start + self.features];
                rmsn_stats("gamma (raw)", gammas);
                rmsn_stats("eps_raw (stored)", eps_raw);
                rmsn_stats("eps_eff = EPS_MIN + exp(eps_raw)", &eps_eff);
                rmsn_stats("mean_sq (per row)", &mean_sq);
                rmsn_stats("x (input)", x);
                rmsn_first("gamma", gammas, 8);
                rmsn_first("eps_raw", eps_raw, 8);
                rmsn_first("eps_eff", &eps_eff, 8);
                rmsn_first("mean_sq", &mean_sq, 8);
                if bad_arg_count > 0 {
                    println!(
                        "    [RMSN] !! WARNING: {} элементов с mean_sq + eps_eff <= 0 \
                         (min_arg = {:.6e}). При <0 sqrt даст NaN, при =0 деление на ноль.",
                        bad_arg_count, min_arg
                    );
                } else {
                    println!(
                        "    [RMSN] min(mean_sq + eps_eff) = {:.6e} (все > 0)",
                        min_arg
                    );
                }
            } else if *RMSN_DEBUG && bad_arg_count > 0 {
                println!(
                    "[RMSN fwd #{}] ANOMALY: {} элементов с mean_sq + eps_eff <= 0 \
                     (min_arg = {:.6e})",
                    call_id, bad_arg_count, min_arg
                );
            }

            // Основной цикл.
            for c in 0..cols {
                let gamma = p[gamma_start + c];
                let eps = eps_eff[c];
                for r in 0..rows {
                    let idx = c * rows + r;
                    let denom = (mean_sq[r] + eps).sqrt();
                    y[idx] = (x[idx] / denom) * gamma;
                }
            }

            if *RMSN_DEBUG {
                let has_anom = rmsn_has_bad(y);
                if log_this || has_anom {
                    if has_anom && !log_this {
                        println!(
                            "[RMSN fwd #{}] ANOMALY (non-finite in output)",
                            call_id
                        );
                    }
                    rmsn_stats("y (output)", y);
                }
                if traj_this {
                    let gammas = &p[gamma_start..gamma_start + self.features];
                    println!(
                        "[RMSN traj fwd #{}] gamma/eps summary:",
                        call_id
                    );
                    rmsn_stats("  gamma", gammas);
                    rmsn_stats("  eps_raw", eps_raw);
                    rmsn_stats("  eps_eff", &eps_eff);
                    let mut min_arg_traj = f32::INFINITY;
                    let mut bad_traj = 0usize;
                    for c in 0..cols {
                        let eps = eps_eff[c];
                        for r in 0..rows {
                            let arg = mean_sq[r] + eps;
                            if arg < min_arg_traj {
                                min_arg_traj = arg;
                            }
                            if arg <= 0.0 {
                                bad_traj += 1;
                            }
                        }
                    }
                    println!(
                        "[RMSN traj fwd #{}] min(mean_sq + eps_eff) = {:.6e}, bad_count = {}",
                        call_id, min_arg_traj, bad_traj
                    );
                }
            }
        });
    }

    // ------------------------------------------------------------------------
    // Backward
    //
    // Формула (фикс B применён):
    //
    //   d_j    = sqrt(mean_sq + ε_j)
    //   gi_c   = go_c · γ_c / d_c  −  (x_c / F) · Σ_j  go_j·γ_j·x_j / d_j³
    //
    //   ∂L/∂γ_c   = Σ_r go_c · x_c / d_c
    //   ∂L/∂ε_c   = Σ_r − 0.5 · go_c · γ_c · x_c / d_c³
    //   ∂L/∂eps_raw_c = ∂L/∂ε_c · exp(eps_raw_c)
    //
    // Куб знаменателя стоит ПОД суммой по j (учёт per-feature ε_j),
    // а не снаружи как d_c³-множитель. При ε_j = const формула совпадает
    // с PyTorch-канонической RMSNorm; при разных ε_j — обобщает её.
    // ------------------------------------------------------------------------

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
            BufferedContext::RMSNormWithLearnableEpsilon { input } => input,
            _ => panic!("Expected RMSNormWithLearnableEpsilon context"),
        };

        let rows = grad_output.rows();
        let cols = grad_output.cols();
        debug_assert_eq!(cols, self.features);
        debug_assert_eq!(rows, input_handle.rows());

        let (log_this, call_id, traj_this) = if *RMSN_DEBUG {
            let n = RMSN_BWD_CALLS.fetch_add(1, Ordering::Relaxed);
            (n < BACKWARD_LOG_LIMIT, n, n % LOG_BWD_TRAJECTORY_EVERY == 0)
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

                let gamma_start = slice.start;
                let eps_start = gamma_start + self.features;

                // mean_sq по строкам (как в forward).
                let mut mean_sq = vec![0.0f32; rows];
                for r in 0..rows {
                    let mut sum_sq = 0.0f32;
                    for c in 0..cols {
                        let idx = c * rows + r;
                        let v = x[idx];
                        sum_sq += v * v;
                    }
                    mean_sq[r] = sum_sq / cols as f32;
                }

                // ε = EPS_MIN + exp(eps_raw); ∂ε/∂eps_raw = exp(eps_raw).
                let eps_raw = &p[eps_start..eps_start + self.features];
                let eps_eff = compute_eps_eff(eps_raw);
                let deps_draw: Vec<f32> = eps_raw.iter().map(|&r| r.exp()).collect();

                let mut grad_gamma = vec![0.0f32; self.features];
                let mut grad_eps_raw = vec![0.0f32; self.features];

                for c in 0..cols {
                    let gamma = p[gamma_start + c];
                    let eps = eps_eff[c];

                    let mut d_gamma_acc = 0.0f32;
                    let mut d_eps_acc = 0.0f32;

                    for r in 0..rows {
                        let idx = c * rows + r;
                        let x_val = x[idx];
                        let gout = go[idx];
                        let denom = (mean_sq[r] + eps).sqrt();
                        let denom3 = denom * denom * denom;

                        // Градиент по входу (фикс B):
                        //   gi_c = go_c·γ_c/d_c − (x_c / F) · Σ_j go_j·γ_j·x_j / d_j³
                        //
                        // В сумме по j для каждого j используется СВОЙ
                        // знаменатель d_j³ (учитывая per-feature ε_j). При
                        // ε_j = const это в точности PyTorch-канон.
                        let term1 = gamma / denom;
                        let sum_j = {
                            let mut s = 0.0f32;
                            for j in 0..cols {
                                let idx_j = j * rows + r;
                                let gamma_j = p[gamma_start + j];
                                let eps_j = eps_eff[j];
                                let denom_j = (mean_sq[r] + eps_j).sqrt();
                                let d3 = denom_j * denom_j * denom_j;
                                s += go[idx_j] * gamma_j * x[idx_j] / d3;
                            }
                            s
                        };
                        let term2 = (1.0 / cols as f32) * x_val * sum_j;
                        gi[idx] = gout * term1 - term2;

                        // Градиенты параметров.
                        //   ∂y_c/∂γ_c = x_c / d_c
                        //   ∂y_c/∂ε_c = − 0.5 · γ_c · x_c / d_c³
                        d_gamma_acc += gout * x_val / denom;
                        d_eps_acc += -0.5 * gout * gamma * x_val / denom3;
                    }

                    grad_gamma[c] = d_gamma_acc;
                    // Chain-rule: ∂L/∂eps_raw = ∂L/∂ε · ∂ε/∂eps_raw.
                    grad_eps_raw[c] = d_eps_acc * deps_draw[c];
                }

                for c in 0..self.features {
                    gp[gamma_start + c] = grad_gamma[c];
                    gp[eps_start + c] = grad_eps_raw[c];
                }

                if *RMSN_DEBUG && log_this {
                    println!(
                        "[RMSN bwd #{}] rows={}, cols={}, slice.start={}",
                        call_id, rows, cols, slice.start
                    );
                    let gammas = &p[gamma_start..gamma_start + self.features];
                    rmsn_stats("gamma", gammas);
                    rmsn_stats("eps_raw", eps_raw);
                    rmsn_stats("eps_eff", &eps_eff);
                    rmsn_stats("x (input)", x);
                    rmsn_stats("go (grad_out)", go);
                    rmsn_stats("grad_gamma", &grad_gamma);
                    rmsn_stats("grad_eps_raw", &grad_eps_raw);
                    rmsn_stats("gi (grad_input)", gi);
                    rmsn_first("grad_gamma", &grad_gamma, 8);
                    rmsn_first("grad_eps_raw", &grad_eps_raw, 8);
                }

                if *RMSN_DEBUG {
                    let has_anom = rmsn_has_bad(&grad_gamma)
                        || rmsn_has_bad(&grad_eps_raw)
                        || rmsn_has_bad(gi);
                    if has_anom && !log_this {
                        println!(
                            "[RMSN bwd #{}] ANOMALY (non-finite grad)",
                            call_id
                        );
                        rmsn_stats("grad_gamma", &grad_gamma);
                        rmsn_stats("grad_eps_raw", &grad_eps_raw);
                        rmsn_stats("gi", gi);
                    }
                }

                if *RMSN_DEBUG && traj_this {
                    let gg_l2: f32 = {
                        let s: f64 = grad_gamma
                            .iter()
                            .filter(|v| v.is_finite())
                            .map(|&v| (v as f64) * (v as f64))
                            .sum();
                        s.sqrt() as f32
                    };
                    let ge_l2: f32 = {
                        let s: f64 = grad_eps_raw
                            .iter()
                            .filter(|v| v.is_finite())
                            .map(|&v| (v as f64) * (v as f64))
                            .sum();
                        s.sqrt() as f32
                    };
                    let gi_l2: f32 = {
                        let s: f64 = gi
                            .iter()
                            .filter(|v| v.is_finite())
                            .map(|&v| (v as f64) * (v as f64))
                            .sum();
                        s.sqrt() as f32
                    };
                    println!(
                        "[RMSN traj bwd #{}] ||grad_gamma||={:.6e}, \
                         ||grad_eps_raw||={:.6e}, ||gi||={:.6e}",
                        call_id, gg_l2, ge_l2, gi_l2
                    );
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