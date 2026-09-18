// src/layers/concrete_dropout/cpu/mod.rs

use std::sync::atomic::{AtomicUsize, Ordering};

use once_cell::sync::Lazy;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::compute_manager::core::dynamic_context::DynamicContext;
use crate::compute_manager::operators_v2::memory_v2::buffer::{MatrixBufferHandle, TempMatrixPool};
use crate::layers::buffered_context::BufferedContext;
use crate::layers::UniversalLayerBuffered;
use crate::model_plan::param_store::ParamSlice;

use super::super::concrete_dropout::ConcreteDropout;

// ============================================================================
// Диагностика ConcreteDropout (CPU)
// ============================================================================
//
// Включается переменной окружения NEUROCORE_DEBUG_CDROPOUT=1.
//
// Логируются:
//   * первые CD_FWD_LOG_LIMIT вызовов forward — статистика logit_p,
//     a (аргумент сигмоиды), z (маска = sigmoid(a)), x (вход), y (выход);
//   * каждый CD_FWD_TRAJ_EVERY-й forward — компактная траектория
//     (logit_p, mean z, L2 y, L2 x, effective_seed);
//   * первые CD_BWD_LOG_LIMIT вызовов backward — статистика go, gi,
//     logit_p, grad_logit_p;
//   * каждый CD_BWD_TRAJ_EVERY-й backward — компактная траектория.
//
// Важно: сами println! выполняются ВНЕ критической секции
// (после выхода из with_cpu_slices_mut), чтобы не блокировать
// MemoryExecutor на время ввода-вывода.

static CD_DEBUG: Lazy<bool> =
    Lazy::new(|| std::env::var("NEUROCORE_DEBUG_CDROPOUT").is_ok());

static CD_FWD_CALLS: Lazy<AtomicUsize> = Lazy::new(|| AtomicUsize::new(0));
static CD_BWD_CALLS: Lazy<AtomicUsize> = Lazy::new(|| AtomicUsize::new(0));

const CD_FWD_LOG_LIMIT: usize = 3;
const CD_BWD_LOG_LIMIT: usize = 3;
const CD_FWD_TRAJ_EVERY: usize = 50;
const CD_BWD_TRAJ_EVERY: usize = 50;

/// Компактная статистика по срезу f32.
#[derive(Debug, Clone, Copy)]
struct CdStats {
    len: usize,
    min: f32,
    max: f32,
    mean: f32,
    l2: f64,
    nan: usize,
    inf: usize,
}

impl CdStats {
    fn empty() -> Self {
        CdStats {
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

fn cd_stats(data: &[f32]) -> CdStats {
    if data.is_empty() {
        return CdStats::empty();
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
    CdStats {
        len: data.len(),
        min: mn,
        max: mx,
        mean,
        l2: sq_sum.sqrt(),
        nan: nan_cnt,
        inf: inf_cnt,
    }
}

fn cd_stats_str(name: &str, s: &CdStats) -> String {
    format!(
        "{}: len={}, min={:.6}, max={:.6}, mean={:.6}, l2={:.6}, nan={}, inf={}",
        name, s.len, s.min, s.max, s.mean, s.l2, s.nan, s.inf
    )
}

#[inline]
fn cd_first(label: &str, data: &[f32], k: usize) -> String {
    let show = data.len().min(k);
    format!("{} (first {}): {:?}", label, show, &data[..show])
}

/// Численно устойчивая сигмоида.
#[inline]
fn cd_sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Снимок внутреннего состояния forward для отложенной печати.
struct CdForwardSnapshot {
    logit_p: f32,
    effective_seed: u64,
    x: Vec<f32>,
    y: Vec<f32>,
    arg: Vec<f32>,
}

/// Снимок внутреннего состояния backward для отложенной печати.
struct CdBackwardSnapshot {
    logit_p: f32,
    x: Vec<f32>,
    go: Vec<f32>,
    gi: Vec<f32>,
    arg: Vec<f32>,
    grad_logit_p: f32,
}

impl UniversalLayerBuffered for ConcreteDropout {
    fn forward_buffered(
        &self,
        input: &MatrixBufferHandle,
        output: &MatrixBufferHandle,
        params: &MatrixBufferHandle,
        slice: &ParamSlice,
        pool: &mut TempMatrixPool,
    ) -> BufferedContext {
        let rows = input.rows();
        let cols = input.cols();
        let total = rows * cols;
        debug_assert!(
            slice.start + self.param_len() <= params.rows() * params.cols(),
            "ConcreteDropout: parameter slice out of bounds"
        );

        // Per-chunk буфер для аргументов сигмоиды.
        // Форма — вектор-столбец длины total.
        let arg_handle = pool.acquire(total, 1);

        let (log_this, call_id, traj_this) = if *CD_DEBUG {
            let n = CD_FWD_CALLS.fetch_add(1, Ordering::Relaxed);
            (n < CD_FWD_LOG_LIMIT, n, n % CD_FWD_TRAJ_EVERY == 0)
        } else {
            (false, 0usize, false)
        };

        // FIX (стохастичность dropout):
        //
        // Раньше RNG создавался с константным seed = self.seed. Это давало
        // одну и ту же последовательность u при каждом forward, а значит —
        // одну и ту же маску z на всех итерациях. Dropout вырождался в
        // фиксированный гейт, и обучение шло крайне медленно.
        //
        // Теперь используется self.next_seed() — атомарный счётчик вызовов
        // в структуре слоя. Каждый forward получает свежий seed, что даёт
        // независимую выборку u. Счётчик общий с GPU-путём, поэтому логика
        // единая.
        let effective_seed = self.next_seed();

        // Снимок для отложенной печати. None — логирование отключено.
        let mut snapshot: Option<CdForwardSnapshot> = None;

        // Всё чтение/запись выполняем одним срезом.
        let ids = [
            input.id(),
            output.id(),
            params.id(),
            arg_handle.id(),
        ];
        input
            .memory()
            .write()
            .unwrap()
            .with_cpu_slices_mut(&ids, |slices| {
                let (first, rest) = slices.split_at_mut(1);
                let x: &[f32] = &*first[0];

                let (second, rest) = rest.split_at_mut(1);
                let y: &mut [f32] = &mut *second[0];

                let (third, rest) = rest.split_at_mut(1);
                let p: &[f32] = &*third[0];

                let (fourth, _) = rest.split_at_mut(1);
                let arg: &mut [f32] = &mut *fourth[0];

                let logit_p = p[slice.start];
                let temp = self.temperature;
                let mut rng = StdRng::seed_from_u64(effective_seed);
                let eps = 1e-8f32;

                for i in 0..total {
                    let u: f32 = rng.gen();
                    let log_u = (u + eps).ln();
                    let log_1mu = (1.0 - u + eps).ln();
                    let a = (logit_p + log_u - log_1mu) / temp;
                    let z = cd_sigmoid(a);
                    arg[i] = a;
                    y[i] = x[i] * z;
                }

                if *CD_DEBUG && (log_this || traj_this) {
                    snapshot = Some(CdForwardSnapshot {
                        logit_p,
                        effective_seed,
                        x: x.to_vec(),
                        y: y.to_vec(),
                        arg: arg.to_vec(),
                    });
                }
            });

        // Печатаем после выхода из замка.
        if let Some(snap) = snapshot {
            let z_vec: Vec<f32> = snap.arg.iter().map(|&a| cd_sigmoid(a)).collect();
            let sx = cd_stats(&snap.x);
            let sy = cd_stats(&snap.y);
            let sa = cd_stats(&snap.arg);
            let sz = cd_stats(&z_vec);
            let z_mean = sz.mean;

            if log_this {
                println!(
                    "[CD fwd #{}] rows={}, cols={}, temp={:.6}, base_seed={}, \
                     effective_seed={}, slice.start={}",
                    call_id,
                    rows,
                    cols,
                    self.temperature,
                    self.seed,
                    snap.effective_seed,
                    slice.start
                );
                println!("    logit_p (raw)      = {:.6}", snap.logit_p);
                println!(
                    "    sigmoid(logit_p)   = {:.6}",
                    cd_sigmoid(snap.logit_p)
                );
                println!("    {}", cd_stats_str("x (input)", &sx));
                println!("    {}", cd_stats_str("y (output)", &sy));
                println!("    {}", cd_stats_str("a (arg)", &sa));
                println!("    {}", cd_stats_str("z (mask)", &sz));
                println!("    {}", cd_first("z", &z_vec, 8));
                println!("    {}", cd_first("y", &snap.y, 8));
            }
            if traj_this {
                println!(
                    "[CD traj fwd #{}] logit_p={:.4} mean_z={:.4} \
                     l2_x={:.4e} l2_y={:.4e} eff_seed={}",
                    call_id, snap.logit_p, z_mean, sx.l2, sy.l2, snap.effective_seed
                );
            }
        }

        BufferedContext::ConcreteDropout {
            input: input.clone(),
            arg: arg_handle,
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
        let (input_handle, arg_handle) = match bc {
            BufferedContext::ConcreteDropout { input, arg } => (input, arg),
            _ => panic!("Expected ConcreteDropout Buffered context"),
        };

        let rows = grad_output.rows();
        let cols = grad_output.cols();
        let total = rows * cols;
        debug_assert_eq!(rows, input_handle.rows());
        debug_assert_eq!(cols, input_handle.cols());
        debug_assert_eq!(total, arg_handle.rows() * arg_handle.cols());
        debug_assert!(
            slice.start + self.param_len() <= params.rows() * params.cols(),
            "ConcreteDropout backward: parameter slice out of bounds"
        );
        debug_assert!(
            slice.start + self.param_len() <= grad_params.rows() * grad_params.cols(),
            "ConcreteDropout backward: grad parameter slice out of bounds"
        );

        let (log_this, call_id, traj_this) = if *CD_DEBUG {
            let n = CD_BWD_CALLS.fetch_add(1, Ordering::Relaxed);
            (n < CD_BWD_LOG_LIMIT, n, n % CD_BWD_TRAJ_EVERY == 0)
        } else {
            (false, 0usize, false)
        };

        let mut snapshot: Option<CdBackwardSnapshot> = None;

        let ids = [
            input_handle.id(),
            grad_output.id(),
            grad_input.id(),
            params.id(),
            grad_params.id(),
            arg_handle.id(),
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

                let (fifth, rest) = rest.split_at_mut(1);
                let gp: &mut [f32] = &mut *fifth[0];

                let (sixth, _) = rest.split_at_mut(1);
                let arg: &[f32] = &*sixth[0];

                let logit_p = p[slice.start];
                let temp = self.temperature;
                let mut grad_logit_p = 0.0f32;

                for i in 0..total {
                    let a = arg[i];
                    let sigmoid = cd_sigmoid(a);
                    let dsigmoid = sigmoid * (1.0 - sigmoid);

                    // Градиент по входу.
                    gi[i] = go[i] * sigmoid;

                    // Градиент по logit_p.
                    grad_logit_p += go[i] * x[i] * dsigmoid / temp;
                }

                // Записываем градиент по logit_p.
                gp[slice.start] = grad_logit_p;

                if *CD_DEBUG && (log_this || traj_this) {
                    snapshot = Some(CdBackwardSnapshot {
                        logit_p,
                        x: x.to_vec(),
                        go: go.to_vec(),
                        gi: gi.to_vec(),
                        arg: arg.to_vec(),
                        grad_logit_p,
                    });
                }
            });

        // Печатаем после выхода из замка.
        if let Some(snap) = snapshot {
            let z_vec: Vec<f32> = snap.arg.iter().map(|&a| cd_sigmoid(a)).collect();
            let sx = cd_stats(&snap.x);
            let sgo = cd_stats(&snap.go);
            let sgi = cd_stats(&snap.gi);
            let sa = cd_stats(&snap.arg);
            let sz = cd_stats(&z_vec);
            let z_mean = sz.mean;

            if log_this {
                println!(
                    "[CD bwd #{}] rows={}, cols={}, temp={:.6}, slice.start={}",
                    call_id, rows, cols, self.temperature, slice.start
                );
                println!("    logit_p (raw)      = {:.6}", snap.logit_p);
                println!("    grad_logit_p       = {:.6e}", snap.grad_logit_p);
                println!("    {}", cd_stats_str("x (input)", &sx));
                println!("    {}", cd_stats_str("go (grad_out)", &sgo));
                println!("    {}", cd_stats_str("gi (grad_input)", &sgi));
                println!("    {}", cd_stats_str("a (arg)", &sa));
                println!("    {}", cd_stats_str("z (mask)", &sz));
                println!("    {}", cd_first("gi", &snap.gi, 8));
            }
            if traj_this {
                println!(
                    "[CD traj bwd #{}] logit_p={:.4} grad_logit_p={:.4e} \
                     mean_z={:.4} l2_go={:.4e} l2_gi={:.4e}",
                    call_id,
                    snap.logit_p,
                    snap.grad_logit_p,
                    z_mean,
                    sgo.l2,
                    sgi.l2
                );
            }
        }
    }

    fn param_len(&self) -> usize {
        1
    }

    fn input_features(&self) -> usize {
        0
    }

    fn output_features(&self) -> usize {
        0
    }
}