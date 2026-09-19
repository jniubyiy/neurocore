// src/layers/learnable_mish/cpu/mod.rs

use std::sync::atomic::{AtomicUsize, Ordering};

use once_cell::sync::Lazy;

use crate::compute_manager::core::dynamic_context::DynamicContext;
use crate::compute_manager::operators_v2::memory_v2::buffer::{MatrixBufferHandle, TempMatrixPool};
use crate::layers::buffered_context::BufferedContext;
use crate::layers::UniversalLayerBuffered;
use crate::model_plan::param_store::ParamSlice;

use super::super::learnable_mish::LearnableMish;

// ============================================================================
// Диагностика LearnableMish (CPU)
// ============================================================================
//
// Включается переменной окружения NEUROCORE_DEBUG_LMISH=1.
//
// Логируются:
//   * первые LMISH_FWD_LOG_LIMIT вызовов forward — статистики x, sp,
//     tanh(λ·sp), y и значение λ;
//   * каждый LMISH_FWD_TRAJ_EVERY-й forward — компактная траектория
//     (λ, mean y, L2 y, L2 x);
//   * первые LMISH_BWD_LOG_LIMIT вызовов backward — статистики x, go, gi,
//     значение λ и накопленный grad_lambda;
//   * каждый LMISH_BWD_TRAJ_EVERY-й backward — компактная траектория.
//
// Дополнительно: при первом forward, если |λ| < LMISH_LAMBDA_WARN_THRESHOLD,
// печатается [LMISH WARN] с гипотезой о вырожденной инициализации λ.
//
// Сами println! выполняются ВНЕ критической секции
// (после выхода из with_cpu_slices_mut), чтобы не блокировать
// MemoryExecutor на время ввода-вывода.

static LMISH_DEBUG: Lazy<bool> =
    Lazy::new(|| std::env::var("NEUROCORE_DEBUG_LMISH").is_ok());

static LMISH_FWD_CALLS: Lazy<AtomicUsize> = Lazy::new(|| AtomicUsize::new(0));
static LMISH_BWD_CALLS: Lazy<AtomicUsize> = Lazy::new(|| AtomicUsize::new(0));
static LMISH_WARN_EMITTED: Lazy<AtomicUsize> = Lazy::new(|| AtomicUsize::new(0));

const LMISH_FWD_LOG_LIMIT: usize = 3;
const LMISH_BWD_LOG_LIMIT: usize = 3;
const LMISH_FWD_TRAJ_EVERY: usize = 50;
const LMISH_BWD_TRAJ_EVERY: usize = 50;

/// Порог |λ|, ниже которого слой считается вырожденным (по гипотезе админа —
/// из-за generic-инициализации около нуля).
const LMISH_LAMBDA_WARN_THRESHOLD: f32 = 0.05;

/// Компактная статистика по срезу f32.
#[derive(Debug, Clone, Copy)]
struct LmStats {
    len: usize,
    min: f32,
    max: f32,
    mean: f32,
    l2: f64,
    nan: usize,
    inf: usize,
}

impl LmStats {
    fn empty() -> Self {
        LmStats {
            len: 0,
            min: f32::NAN,
            max: f32::NAN,
            mean: f32::NAN,
            l2: 0.0,
            nan: 0,
            inf: 0,
        }
    }
}

fn lm_stats(data: &[f32]) -> LmStats {
    if data.is_empty() {
        return LmStats::empty();
    }
    let mut mn = f32::INFINITY;
    let mut mx = f32::NEG_INFINITY;
    let mut sum = 0.0f64;
    let mut sq_sum = 0.0f64;
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
        if v < mn {
            mn = v;
        }
        if v > mx {
            mx = v;
        }
        sum += v as f64;
        sq_sum += (v as f64) * (v as f64);
    }
    let finite = data.len().saturating_sub(nan_cnt + inf_cnt);
    let mean = if finite > 0 {
        (sum / finite as f64) as f32
    } else {
        f32::NAN
    };
    LmStats {
        len: data.len(),
        min: mn,
        max: mx,
        mean,
        l2: sq_sum.sqrt(),
        nan: nan_cnt,
        inf: inf_cnt,
    }
}

fn lm_stats_str(name: &str, s: &LmStats) -> String {
    format!(
        "{}: len={}, min={:.6}, max={:.6}, mean={:.6}, l2={:.6}, nan={}, inf={}",
        name, s.len, s.min, s.max, s.mean, s.l2, s.nan, s.inf
    )
}

#[inline]
fn lm_first(label: &str, data: &[f32], k: usize) -> String {
    let show = data.len().min(k);
    format!("{} (first {}): {:?}", label, show, &data[..show])
}

/// Численно устойчивый softplus.
#[inline]
fn softplus_stable(z: f32) -> f32 {
    if z > 0.0 {
        z + (-z).exp().ln_1p()
    } else {
        z.exp().ln_1p()
    }
}

/// Снимок forward для отложенной печати.
struct LmForwardSnapshot {
    lambda: f32,
    x: Vec<f32>,
    sp: Vec<f32>,
    tanh_lambda_sp: Vec<f32>,
    y: Vec<f32>,
}

/// Снимок backward для отложенной печати.
struct LmBackwardSnapshot {
    lambda: f32,
    grad_lambda: f32,
    x: Vec<f32>,
    go: Vec<f32>,
    gi: Vec<f32>,
}

impl UniversalLayerBuffered for LearnableMish {
    fn forward_buffered(
        &self,
        input: &MatrixBufferHandle,
        output: &MatrixBufferHandle,
        params: &MatrixBufferHandle,
        slice: &ParamSlice,
        _pool: &mut TempMatrixPool,
    ) -> BufferedContext {
        let cols = input.cols();
        debug_assert_eq!(cols, self.features);
        debug_assert!(
            slice.start + self.param_len() <= params.rows() * params.cols(),
            "LearnableMish: parameter slice out of bounds"
        );

        let (log_this, call_id, traj_this) = if *LMISH_DEBUG {
            let n = LMISH_FWD_CALLS.fetch_add(1, Ordering::Relaxed);
            (n < LMISH_FWD_LOG_LIMIT, n, n % LMISH_FWD_TRAJ_EVERY == 0)
        } else {
            (false, 0usize, false)
        };

        // Снимок для отложенной печати. None — логирование отключено.
        let mut snapshot: Option<LmForwardSnapshot> = None;

        let ids = [input.id(), output.id(), params.id()];
        input.memory().write().unwrap().with_cpu_slices_mut(&ids, |slices| {
            let (first, rest) = slices.split_at_mut(1);
            let x: &[f32] = &*first[0];
            let (second, rest) = rest.split_at_mut(1);
            let y: &mut [f32] = &mut *second[0];
            let (third, _) = rest.split_at_mut(1);
            let p: &[f32] = &*third[0];

            let lambda = p[slice.start];

            for i in 0..x.len() {
                let x_val = x[i];
                let sp = softplus_stable(x_val);
                let tanh_sp = (lambda * sp).tanh();
                y[i] = x_val * tanh_sp;
            }

            if *LMISH_DEBUG && (log_this || traj_this) {
                let mut sp_vec = Vec::with_capacity(x.len());
                let mut tls_vec = Vec::with_capacity(x.len());
                for &x_val in x.iter() {
                    let sp = softplus_stable(x_val);
                    sp_vec.push(sp);
                    tls_vec.push((lambda * sp).tanh());
                }
                snapshot = Some(LmForwardSnapshot {
                    lambda,
                    x: x.to_vec(),
                    sp: sp_vec,
                    tanh_lambda_sp: tls_vec,
                    y: y.to_vec(),
                });
            }
        });

        // Печатаем после выхода из замка.
        if let Some(snap) = snapshot {
            let sx = lm_stats(&snap.x);
            let ssp = lm_stats(&snap.sp);
            let stls = lm_stats(&snap.tanh_lambda_sp);
            let sy = lm_stats(&snap.y);

            // Однократное предупреждение о вырожденной инициализации λ.
            if snap.lambda.abs() < LMISH_LAMBDA_WARN_THRESHOLD
                && LMISH_WARN_EMITTED
                    .compare_exchange(0, 1, Ordering::Relaxed, Ordering::Relaxed)
                    .is_ok()
            {
                println!(
                    "[LMISH WARN] |λ|={:.6e} < {:.3e}. Слой почти обнуляет выход: \
                     tanh(λ·sp)≈λ·sp≈0 ⇒ y≈0. Проверь инициализацию λ \
                     (bridge_v2 / generic RandomUniform даёт λ≈0).",
                    snap.lambda, LMISH_LAMBDA_WARN_THRESHOLD
                );
            }

            if log_this {
                println!(
                    "[LMISH fwd #{}] features={}, slice.start={}",
                    call_id, cols, slice.start
                );
                println!("    λ                  = {:.6e}", snap.lambda);
                println!("    {}", lm_stats_str("x (input)", &sx));
                println!("    {}", lm_stats_str("sp (softplus)", &ssp));
                println!(
                    "    {}",
                    lm_stats_str("tanh(λ·sp)", &stls)
                );
                println!("    {}", lm_stats_str("y (output)", &sy));
                println!("    {}", lm_first("y", &snap.y, 8));
            }
            if traj_this {
                println!(
                    "[LMISH traj fwd #{}] λ={:.6e} mean_y={:.6} l2_x={:.4e} l2_y={:.4e}",
                    call_id, snap.lambda, sy.mean, sx.l2, sy.l2
                );
            }
        }

        BufferedContext::LearnableMish {
            input: input.clone(),
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
        let input_handle = match bc {
            BufferedContext::LearnableMish { input } => input,
            _ => panic!("Expected LearnableMish context"),
        };

        let rows = grad_output.rows();
        let cols = grad_output.cols();
        debug_assert_eq!(cols, self.features);
        debug_assert_eq!(rows, input_handle.rows());
        debug_assert!(
            slice.start + self.param_len() <= params.rows() * params.cols(),
            "LearnableMish backward: parameter slice out of bounds"
        );
        debug_assert!(
            slice.start + self.param_len() <= grad_params.rows() * grad_params.cols(),
            "LearnableMish backward: grad parameter slice out of bounds"
        );

        let (log_this, call_id, traj_this) = if *LMISH_DEBUG {
            let n = LMISH_BWD_CALLS.fetch_add(1, Ordering::Relaxed);
            (n < LMISH_BWD_LOG_LIMIT, n, n % LMISH_BWD_TRAJ_EVERY == 0)
        } else {
            (false, 0usize, false)
        };

        let mut snapshot: Option<LmBackwardSnapshot> = None;

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

                let lambda = p[slice.start];
                let mut grad_lambda = 0.0f32;

                for i in 0..x.len() {
                    let x_val = x[i];
                    let sp = softplus_stable(x_val);
                    let tanh_sp = (lambda * sp).tanh();
                    let dtanh = 1.0 - tanh_sp * tanh_sp;
                    let sigmoid = 1.0 / (1.0 + (-x_val).exp());

                    // ∂y/∂x = tanh(λ·sp) + x · dtanh · λ · sigmoid(x)
                    let dx = tanh_sp + x_val * dtanh * lambda * sigmoid;
                    gi[i] = go[i] * dx;

                    // ∂y/∂λ = x · dtanh · sp
                    grad_lambda += go[i] * x_val * dtanh * sp;
                }

                gp[slice.start] = grad_lambda;

                if *LMISH_DEBUG && (log_this || traj_this) {
                    snapshot = Some(LmBackwardSnapshot {
                        lambda,
                        grad_lambda,
                        x: x.to_vec(),
                        go: go.to_vec(),
                        gi: gi.to_vec(),
                    });
                }
            });

        if let Some(snap) = snapshot {
            let sx = lm_stats(&snap.x);
            let sgo = lm_stats(&snap.go);
            let sgi = lm_stats(&snap.gi);

            if log_this {
                println!(
                    "[LMISH bwd #{}] features={}, slice.start={}",
                    call_id, cols, slice.start
                );
                println!("    λ                  = {:.6e}", snap.lambda);
                println!("    grad_lambda        = {:.6e}", snap.grad_lambda);
                println!("    {}", lm_stats_str("x (input)", &sx));
                println!("    {}", lm_stats_str("go (grad_out)", &sgo));
                println!("    {}", lm_stats_str("gi (grad_input)", &sgi));
                println!("    {}", lm_first("gi", &snap.gi, 8));
            }
            if traj_this {
                println!(
                    "[LMISH traj bwd #{}] λ={:.6e} grad_λ={:.4e} l2_go={:.4e} l2_gi={:.4e}",
                    call_id, snap.lambda, snap.grad_lambda, sgo.l2, sgi.l2
                );
            }
        }
    }

    fn param_len(&self) -> usize {
        1
    }

    fn input_features(&self) -> usize {
        self.features
    }

    fn output_features(&self) -> usize {
        self.features
    }
}