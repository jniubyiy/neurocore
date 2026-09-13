// src/plans/training_plan/execution.rs

use std::collections::HashMap;
use std::path::PathBuf;
use std::thread;
use std::time::Instant;

use once_cell::sync::Lazy;

use rand::Rng;
use rand::SeedableRng;

use crate::compute_manager::dim_change::DynamicTensor;
use crate::compute_manager::graph::model::MixedModel;
use crate::compute_manager::graph::types::Model;
use crate::device_plan::DevicePlan;
use crate::logging::training_monitor::TrainingMonitor;
use crate::model_plan::Plan;
use crate::tensor::Tensor2D;

use super::plan::{Initializer, TrainingPlan};
use super::profiling::{Profiler, ProfileMode, ProfileResult};
use crate::compute_manager::memory_executor::types::MemoryDeviceKind;

// ============================================================================
// Отладочные переключатели
// ============================================================================
//
// NEUROCORE_DEBUG_TRAIN=1  — включает диагностику тренировочного цикла:
//   * шапка (num_samples, lr, init ||p||);
//   * loss и ||g|| по каждому батчу (первые LOG_FIRST_EPOCHS эпох);
//   * содержимое и per-sample MSE всех батчей на epoch 0;
//   * ANOMALY-эпохи (loss батча > ANOMALY_MULTIPLIER × avg) — полный список
//     батчей, per-sample MSE, содержимое худшего sample'а;
//   * эволюция β и θ слоя LearnableSoftplus из ParamStore
//     (первые LOG_LSF_EVO_FIRST эпох, затем раз в LOG_LSF_EVO_EVERY эпох);
//   * раз в LOG_NORM_EVERY эпох — ||params||_2.
//
// NEUROCORE_DEBUG_LAYERS=1 — точечная диагностика для сравнения CPU и GPU:
//   * после forward — checksum prediction (sum/l2/min/max);
//   * после backward — checksum первого gradient-буфера (sum/l2/min/max);
//   * после update (для последнего батча первых эпох) — ||params||_2.
//   Печатается для каждого батча всех эпох. Логи одинакового формата
//   для CPU и GPU, что позволяет построчное сравнение.
//
// ВАЖНО: чтение параметров для checksum выполняется через
// `ParamGradients::to_flat_vec` (для градиентов) и через
// `Tensor`-безопасный путь (для prediction). Для параметров используется
// `param_store_read_all_safe`, который корректно обрабатывает GPU-буферы:
// скачивает их во временный CPU-хэндл и читает оттуда.
// ============================================================================

static TRAIN_DEBUG: Lazy<bool> =
    Lazy::new(|| std::env::var("NEUROCORE_DEBUG_TRAIN").is_ok());

static LAYER_DEBUG: Lazy<bool> =
    Lazy::new(|| std::env::var("NEUROCORE_DEBUG_LAYERS").is_ok());

const LOG_FIRST_EPOCHS: usize = 3;
const LOG_NORM_EVERY: usize = 10;
const ANOMALY_MULTIPLIER: f32 = 2.0;
const ANOMALY_MIN_LOSS: f32 = 0.05;
const ANOMALY_DIAG_LIMIT: usize = 10;

// Диагностика эволюции β/θ
const LOG_LSF_EVO_FIRST: usize = 20;
const LOG_LSF_EVO_EVERY: usize = 20;

// ----------------------------------------------------------------------------
// Утилиты
// ----------------------------------------------------------------------------

fn train_dbg_l2(data: &[f32]) -> f64 {
    let mut s = 0.0f64;
    for &v in data {
        if v.is_finite() {
            s += (v as f64) * (v as f64);
        }
    }
    s.sqrt()
}

fn train_dbg_print_stats(name: &str, data: &[f32]) {
    if data.is_empty() {
        println!("    [TRAIN] {}: <empty>", name);
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
        "    [TRAIN] {}: len={}, min={:.6}, max={:.6}, mean={:.6}, l2={:.6}, nan={}, inf={}",
        name, data.len(), mn, mx, mean, train_dbg_l2(data), nan_cnt, inf_cnt
    );
}

fn train_dbg_print_vec(label: &str, data: &[f32]) {
    println!("    [TRAIN] {} (len={}): {:?}", label, data.len(), data);
}

/// Краткая однострочная сводка по срезу для сравнения CPU/GPU.
fn train_dbg_summary(data: &[f32]) -> String {
    if data.is_empty() {
        return "len=0".to_string();
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
    let l2 = train_dbg_l2(data);
    format!(
        "len={} sum={:.6e} l2={:.6e} min={:.6e} max={:.6e} nan={} inf={}",
        data.len(), sum, l2, mn, mx, nan_cnt, inf_cnt
    )
}

/// Безопасное чтение всех параметров модели.
///
/// В отличие от `ParamStore::get_all_params`, корректно обрабатывает
/// GPU-буферы: скачивает их во временный CPU-хэндл и читает оттуда.
/// Работает только при наличии активного GpuCompute — который есть у
/// `MixedModel` через `compute_executor`.
fn read_all_params_safe(model: &MixedModel) -> Option<Vec<f32>> {
    let ps = model.param_store().lock().unwrap();
    if ps.is_empty() {
        return Some(Vec::new());
    }
    let gpu_opt = model.compute_executor().gpu_compute();
    let mut result = Vec::with_capacity(ps.total_params());
    for buffer_idx in 0..ps.num_buffers() {
        let buf = ps.get_param_buffer_by_idx(buffer_idx);
        if buf.params.is_gpu() {
            let gpu = gpu_opt.as_ref()?;
            let data = gpu.download_gpu_handle_to_vec(&buf.params);
            result.extend_from_slice(&data);
        } else {
            let guard = buf.params.read();
            let slice = guard.as_slice()?;
            result.extend_from_slice(slice);
        }
    }
    Some(result)
}

fn per_sample_mse(pred_flat: &[f32], target_flat: &[f32]) -> f32 {
    assert_eq!(pred_flat.len(), target_flat.len());
    let n = pred_flat.len();
    if n == 0 { return 0.0; }
    let mut s = 0.0f32;
    for i in 0..n {
        let d = pred_flat[i] - target_flat[i];
        s += d * d;
    }
    s / n as f32
}

// ----------------------------------------------------------------------------
// Диагностика эволюции β и θ слоя LearnableSoftplus
// ----------------------------------------------------------------------------

struct LsfEvoSnap {
    model_idx: usize,
    layer_idx: usize,
    features: usize,
    raw_betas: Vec<f32>,
    thetas: Vec<f32>,
    grad_raw_betas: Vec<f32>,
    grad_thetas: Vec<f32>,
}

/// Собирает снимок raw_beta/theta и их градиентов для всех LearnableSoftplus
/// в модели. Работает только на CPU-параметрах (этот путь активируется
/// через TRAIN_DEBUG и рассчитан на отладку CPU-сценариев).
fn collect_lsf_evo(model: &MixedModel) -> Vec<LsfEvoSnap> {
    let ps_guard = model.param_store().lock().unwrap();
    let mut out = Vec::new();
    for (m_idx, m) in model.models().iter().enumerate() {
        if let Model::UniversalProcessor(layers, slices, _) = m {
            for (l_idx, layer) in layers.iter().enumerate() {
                if let Some(lsf) = layer.as_learnable_softplus() {
                    let f = lsf.features;
                    let slice = &slices[l_idx];
                    let params_h = ps_guard.params_handle(slice);
                    let grads_h = ps_guard.grads_handle(slice);
                    if params_h.is_gpu() || grads_h.is_gpu() {
                        // Пропускаем GPU-параметры: этот диагностический путь
                        // рассчитан на CPU.
                        continue;
                    }
                    let raw_betas = params_h.read_range(slice.start, f);
                    let thetas = params_h.read_range(slice.start + f, f);
                    let grad_raw_betas = grads_h.read_range(slice.start, f);
                    let grad_thetas = grads_h.read_range(slice.start + f, f);
                    out.push(LsfEvoSnap {
                        model_idx: m_idx,
                        layer_idx: l_idx,
                        features: f,
                        raw_betas,
                        thetas,
                        grad_raw_betas,
                        grad_thetas,
                    });
                }
            }
        }
    }
    out
}

/// Печатает эволюцию β/θ для текущей эпохи.
fn print_lsf_evo(epoch: usize, snaps: &[LsfEvoSnap]) {
    if snaps.is_empty() {
        println!("  [LSF-EVOLUTION ep{}] (no LearnableSoftplus on CPU found)", epoch);
        return;
    }
    for s in snaps {
        let mut beta_min = f32::INFINITY;
        let mut beta_max = f32::NEG_INFINITY;
        let mut beta_sum = 0.0f32;
        let mut beta_dev_inf = 0.0f32;
        let mut n_valid = 0usize;
        for &rb in &s.raw_betas {
            let b = (1.0 + rb).max(1e-3);
            if b < beta_min { beta_min = b; }
            if b > beta_max { beta_max = b; }
            beta_sum += b;
            let dev = (b - 1.0).abs();
            if dev > beta_dev_inf { beta_dev_inf = dev; }
            n_valid += 1;
        }
        let beta_mean = if n_valid > 0 { beta_sum / n_valid as f32 } else { 0.0 };

        let mut theta_min = f32::INFINITY;
        let mut theta_max = f32::NEG_INFINITY;
        let mut theta_sum = 0.0f32;
        let mut theta_abs_inf = 0.0f32;
        for &t in &s.thetas {
            if t < theta_min { theta_min = t; }
            if t > theta_max { theta_max = t; }
            theta_sum += t;
            let a = t.abs();
            if a > theta_abs_inf { theta_abs_inf = a; }
        }
        let theta_mean = if !s.thetas.is_empty() {
            theta_sum / s.thetas.len() as f32
        } else {
            0.0
        };

        let grad_beta_l2 = train_dbg_l2(&s.grad_raw_betas);
        let grad_theta_l2 = train_dbg_l2(&s.grad_thetas);
        let grad_beta_inf = s
            .grad_raw_betas
            .iter()
            .map(|v| v.abs())
            .fold(0.0, f32::max);
        let grad_theta_inf = s
            .grad_thetas
            .iter()
            .map(|v| v.abs())
            .fold(0.0, f32::max);

        println!(
            "  [LSF-EVOLUTION ep{}] m{} l{} F={}",
            epoch, s.model_idx, s.layer_idx, s.features
        );
        println!(
            "    β: min={:.6} max={:.6} mean={:.6} |β−1|_inf={:.6}",
            beta_min, beta_max, beta_mean, beta_dev_inf
        );
        println!(
            "    θ: min={:.6} max={:.6} mean={:.6} |θ|_inf={:.6}",
            theta_min, theta_max, theta_mean, theta_abs_inf
        );
        println!(
            "    grad_raw_β: l2={:.6} |·|_inf={:.6}",
            grad_beta_l2, grad_beta_inf
        );
        println!(
            "    grad_θ:     l2={:.6} |·|_inf={:.6}",
            grad_theta_l2, grad_theta_inf
        );
    }
}

// ----------------------------------------------------------------------------
// Структуры
// ----------------------------------------------------------------------------

struct BatchInfo {
    start: usize,
    end: usize,
    loss: f32,
    grad_l2: f64,
}

pub struct TrainingResult {
    pub tensors: HashMap<String, DynamicTensor>,
    pub final_loss: f32,
    pub training_time_secs: f64,
    pub best_epoch: usize,
    pub best_loss: f32,
    pub zero_loss_epoch: Option<usize>,
    pub profile: Option<ProfileResult>,
    pub monitor_summary: Option<crate::logging::TrainingSummary>,
}

// ----------------------------------------------------------------------------
// Публичный API
// ----------------------------------------------------------------------------

pub fn execute(plan: &TrainingPlan, device_plan: &DevicePlan) -> Result<TrainingResult, String> {
    let plan = plan.clone();
    let device_plan = device_plan.clone();

    let handle = thread::Builder::new()
        .stack_size(32 * 1024 * 1024)
        .spawn(move || {
            let model_desc = (plan.model_fn)();
            let _ = Plan::from_layer_descs(model_desc.clone())?;
            let mut model = MixedModel::from_plan_with_device_plan(
                model_desc,
                device_plan.clone(),
            )?;
            execute_inner(&plan, &device_plan, &mut model)
        })
        .map_err(|e| format!("Failed to spawn training thread: {}", e))?;

    match handle.join() {
        Ok(inner_result) => inner_result,
        Err(_) => Err("Training thread panicked".to_string()),
    }
}

// ----------------------------------------------------------------------------
// Основная функция
// ----------------------------------------------------------------------------

fn execute_inner(
    plan: &TrainingPlan,
    _device_plan: &DevicePlan,
    model: &mut MixedModel,
) -> Result<TrainingResult, String> {
    let start_time = Instant::now();

    if plan.train_data_streams.is_some() || plan.target_data_streams.is_some() ||
       plan.test_input_streams.is_some() || plan.test_target_streams.is_some() {
        return Err(
            "Многопотоковые данные (train_data_streams, target_data_streams, \
             test_input_streams, test_target_streams) пока не поддерживаются \
             в автоматическом обучении. Используйте ручной цикл с forward_multi/backward_multi."
                .to_string(),
        );
    }

    {
        let mut ps = model.param_store().lock().unwrap();
        let len = ps.total_params();
        match &plan.initializer {
            Initializer::Zeros => ps.set_all_params(&vec![0.0f32; len]),
            Initializer::Ones => ps.set_all_params(&vec![1.0f32; len]),
            Initializer::RandomUniform { min, max } => {
                let mut rng: Box<dyn rand::RngCore> = if let Some(seed) = plan.seed {
                    Box::new(rand::rngs::StdRng::seed_from_u64(seed))
                } else {
                    Box::new(rand::thread_rng())
                };
                let mut params = vec![0.0f32; len];
                for p in &mut params {
                    *p = rng.gen_range(*min..*max);
                }
                ps.set_all_params(&params);
            }
        }
    }

    let opt_chain = plan.optimizer_desc.build_chain();

    let learning_rate = opt_chain
        .cubes()
        .iter()
        .find_map(|cube| {
            cube.as_any()
                .downcast_ref::<crate::optimizers::scale_gradient::ScaleGradient>()
                .map(|sg| sg.factor)
        })
        .unwrap_or(0.01);

    let mut monitor = if plan.monitoring {
        let dump_dir = PathBuf::from("nan_dumps");
        let _ = std::fs::create_dir_all(&dump_dir);
        Some(TrainingMonitor::new(
            plan.monitor_config.clone(),
            learning_rate,
            dump_dir,
        ))
    } else {
        None
    };

    let train_data = match &plan.train_data {
        Some(data) => data.clone(),
        None => return Err("Training data not provided".into()),
    };
    let target_data = match &plan.target_data {
        Some(data) => data.clone(),
        None => train_data.clone(),
    };
    assert_eq!(
        train_data.num_samples(),
        target_data.num_samples(),
        "Training data and target data must have the same number of samples"
    );

    let num_samples = train_data.num_samples();
    let batch_size = plan.batch_size.max(1);
    let num_batches_per_epoch = (num_samples + batch_size - 1) / batch_size;

    let mut profiler = if plan.profile != ProfileMode::None {
        Some(Profiler::new(plan.profile))
    } else {
        None
    };

    // Проверяем, являются ли параметры CPU (для TRAIN_DEBUG-путей, которые
    // читают параметры как CPU).
    let params_are_cpu_initially = {
        let ps = model.param_store().lock().unwrap();
        ps.is_empty() || {
            let buf = ps.get_param_buffer_by_idx(0);
            !buf.params.is_gpu()
        }
    };

    if *TRAIN_DEBUG && params_are_cpu_initially {
        println!("=== [TRAIN DEBUG] execution_inner ===");
        println!("  num_samples = {}", num_samples);
        println!("  batch_size  = {}", batch_size);
        println!("  epochs      = {}", plan.epochs);
        println!("  lr (from ScaleGradient) = {:.6}", learning_rate);
        println!("  num mini-batches/epoch  = {}", num_batches_per_epoch);
        if let Some(flat) = read_all_params_safe(model) {
            println!("  total params = {}", flat.len());
            train_dbg_print_stats("init params", &flat);
        }
        let snaps = collect_lsf_evo(model);
        print_lsf_evo(0, &snaps);
        println!();
    }

    if *LAYER_DEBUG {
        println!("=== [LAYERS DEBUG] per-batch checksums ===");
        println!(
            "  num_samples={} batch_size={} num_batches/epoch={}",
            num_samples, batch_size, num_batches_per_epoch
        );
        if let Some(flat) = read_all_params_safe(model) {
            println!("  init_params: {}", train_dbg_summary(&flat));
        } else {
            println!("  init_params: (safe read returned None)");
        }
        println!();
    }

    let mut best_loss = f32::MAX;
    let mut best_epoch = 0usize;
    let mut zero_loss_epoch: Option<usize> = None;
    let mut anomaly_diag_done: usize = 0;

    for epoch in 0..plan.epochs {
        model.compute_executor().redistribute(model.models(), plan.batch_size, true);
        let placement = model.compute_executor().get_placement();
        model.migrate_parameters(&placement)?;

        let mut epoch_loss = 0.0f32;

        let log_all_batches_this_epoch = *TRAIN_DEBUG && epoch < LOG_FIRST_EPOCHS;
        let log_epoch_summary_this_epoch =
            *TRAIN_DEBUG && (epoch < LOG_FIRST_EPOCHS || epoch % LOG_NORM_EVERY == 0);

        let mut epoch_batches: Vec<BatchInfo> = Vec::with_capacity(num_batches_per_epoch);

        for (batch_idx, start) in (0..num_samples).step_by(batch_size).enumerate() {
            let end = (start + batch_size).min(num_samples);
            let batch_size_actual = end - start;

            let batch_tensor = train_data.batch(start, end);
            let target_batch = target_data.batch(start, end);

            if *TRAIN_DEBUG && epoch == 0 {
                let input_flat = batch_tensor.to_flat();
                let target_flat = target_batch.to_flat();
                println!(
                    "  [DATA e0 b{}-{}] input dims = {:?}, target dims = {:?}",
                    start, end,
                    train_data.dimensions(),
                    target_data.dimensions(),
                );
                train_dbg_print_vec(&format!("input b{}-{}", start, end), &input_flat);
                train_dbg_print_vec(&format!("target b{}-{}", start, end), &target_flat);
                train_dbg_print_stats(&format!("input b{}-{}", start, end), &input_flat);

                println!(
                    "    [TRAIN] per-sample MSE for batch [{}-{}) on epoch 0:",
                    start, end
                );
                for k in 0..batch_size_actual {
                    let s = start + k;
                    let e = s + 1;
                    let sub_input = train_data.batch(s, e);
                    let sub_target = target_data.batch(s, e);
                    let (sub_pred, _) = model.forward(sub_input);
                    let pv = sub_pred.to_flat();
                    let tv = sub_target.to_flat();
                    let loss = per_sample_mse(&pv, &tv);
                    println!("      sample {:>2}: MSE = {:.6}", s, loss);
                }
                println!();
            }

            let t0 = Instant::now();
            let (pred, _ctxs) = model.forward(batch_tensor.clone());
            let forward_dt = t0.elapsed().as_nanos() as u64;

            // === Точечная диагностика: checksum prediction ===
            if *LAYER_DEBUG {
                let pred_flat = pred.to_flat();
                println!(
                    "[DBG-PRED] epoch={:>3} batch={:>2} [{}..{}) {}",
                    epoch, batch_idx, start, end, train_dbg_summary(&pred_flat)
                );
            }

            let t1 = Instant::now();
            let (loss, delta) = model.compute_loss(
                plan.loss_desc.clone(),
                &pred,
                &target_batch,
            );
            let loss_dt = t1.elapsed().as_nanos() as u64;

            let t2 = Instant::now();
            let (_, grads) = model.backward(delta);
            let backward_dt = t2.elapsed().as_nanos() as u64;

            let grads_flat = grads.to_flat_vec();
            let grad_l2 = train_dbg_l2(&grads_flat);

            // === Точечная диагностика: checksum градиентов ===
            if *LAYER_DEBUG {
                println!(
                    "[DBG-GRAD] epoch={:>3} batch={:>2} [{}..{}) {}",
                    epoch, batch_idx, start, end, train_dbg_summary(&grads_flat)
                );
            }

            epoch_batches.push(BatchInfo {
                start,
                end,
                loss,
                grad_l2,
            });

            if *TRAIN_DEBUG && log_all_batches_this_epoch {
                println!(
                    "  [e{} b{}-{}] loss={:.6}  |grad|={:.6}  (fwd={}us loss={}us bwd={}us)",
                    epoch, start, end, loss, grad_l2,
                    forward_dt / 1000, loss_dt / 1000, backward_dt / 1000
                );
                if epoch == 0 && start == 0 {
                    train_dbg_print_stats("grad (first batch)", &grads_flat);
                }
            }

            let t3 = Instant::now();
            model.update_params_buffered(plan.optimizer_desc.clone(), &[]);
            let update_dt = t3.elapsed().as_nanos() as u64;

            // === Точечная диагностика: checksum параметров после update ===
            // Для первых эпох, после последнего батча. Безопасно читает
            // параметры как на CPU, так и на GPU (через временный CPU-хэндл).
            if *LAYER_DEBUG && batch_idx == num_batches_per_epoch - 1 && epoch < LOG_FIRST_EPOCHS {
                if let Some(flat) = read_all_params_safe(model) {
                    println!(
                        "[DBG-PARAMS-AFTER-EPOCH] epoch={:>3} {}",
                        epoch, train_dbg_summary(&flat)
                    );
                } else {
                    println!(
                        "[DBG-PARAMS-AFTER-EPOCH] epoch={:>3} (safe read returned None)",
                        epoch
                    );
                }
            }

            epoch_loss += loss * batch_size_actual as f32;

            if let Some(ref mut mon) = monitor {
                mon.record_step(loss, Some(&grads_flat), None);
            }

            if let Some(ref mut prof) = profiler {
                if prof.mode == ProfileMode::Time || prof.mode == ProfileMode::Full {
                    prof.record_timing(0, "batch", "CPU", "forward", forward_dt);
                    prof.record_timing(0, "batch", "CPU", "loss", loss_dt);
                    prof.record_timing(0, "batch", "CPU", "backward", backward_dt);
                    prof.record_timing(0, "batch", "CPU", "update", update_dt);
                }
                if prof.mode == ProfileMode::Memory || prof.mode == ProfileMode::Full {
                    let mem = model.memory_executor();
                    let me = mem.read().unwrap();
                    let kinds = [
                        MemoryDeviceKind::HostRam,
                        MemoryDeviceKind::SsdCache,
                    ];
                    for kind in &kinds {
                        let used = me.current_usage(*kind);
                        prof.record_memory(0, &format!("{:?}", kind), "batch", used, used);
                    }
                }
            }
        }

        let avg_loss = if num_samples > 0 {
            epoch_loss / num_samples as f32
        } else {
            0.0
        };

        // ====================================================================
        // Диагностика эволюции β/θ
        // ====================================================================
        if *TRAIN_DEBUG {
            let log_evo = epoch < LOG_LSF_EVO_FIRST
                || epoch % LOG_LSF_EVO_EVERY == 0
                || epoch == plan.epochs - 1;
            if log_evo {
                let snaps = collect_lsf_evo(model);
                print_lsf_evo(epoch, &snaps);
            }
        }

        // ====================================================================
        // Диагностика аномалий в эпохе
        // ====================================================================
        if *TRAIN_DEBUG && !epoch_batches.is_empty() {
            let n = epoch_batches.len();
            let sum: f32 = epoch_batches.iter().map(|b| b.loss).sum();
            let avg = sum / n as f32;

            let anomalous: Vec<&BatchInfo> = epoch_batches
                .iter()
                .filter(|b| b.loss > ANOMALY_MULTIPLIER * avg && b.loss > ANOMALY_MIN_LOSS)
                .collect();

            if !anomalous.is_empty() {
                println!(
                    "=== [ANOMALY EPOCH {}] avg={:.6}, anomalous batches: {} ===",
                    epoch, avg, anomalous.len()
                );
                for b in &epoch_batches {
                    let is_anom = b.loss > ANOMALY_MULTIPLIER * avg && b.loss > ANOMALY_MIN_LOSS;
                    println!(
                        "  b{}-{}: loss={:.6}  |grad|={:.6}{}",
                        b.start, b.end, b.loss, b.grad_l2,
                        if is_anom { "  <-- ANOMALY" } else { "" }
                    );
                }

                if anomaly_diag_done < ANOMALY_DIAG_LIMIT {
                    anomaly_diag_done += 1;
                    for b in anomalous {
                        println!(
                            "  [per-sample diag] epoch {} batch b{}-{}:",
                            epoch, b.start, b.end
                        );
                        let mut per_sample: Vec<(usize, f32)> = Vec::with_capacity(b.end - b.start);
                        for s in b.start..b.end {
                            let sub_input = train_data.batch(s, s + 1);
                            let sub_target = target_data.batch(s, s + 1);
                            let (sub_pred, _) = model.forward(sub_input);
                            let pv = sub_pred.to_flat();
                            let tv = sub_target.to_flat();
                            let loss = per_sample_mse(&pv, &tv);
                            per_sample.push((s, loss));
                        }
                        let (_, &(max_s, _)) = per_sample
                            .iter()
                            .enumerate()
                            .max_by(|a, bb| a.1 .1.partial_cmp(&bb.1 .1).unwrap_or(std::cmp::Ordering::Equal))
                            .unwrap_or((0, &(b.start, 0.0)));
                        for &(s, loss) in &per_sample {
                            let mark = if s == max_s { "  <-- MAX" } else { "" };
                            println!("    sample {:>3}: MSE = {:.6}{}", s, loss, mark);
                        }
                        let worst_input = train_data.batch(max_s, max_s + 1);
                        let worst_target = target_data.batch(max_s, max_s + 1);
                        println!(
                            "    worst sample {:>3}: input={:?}  target={:?}",
                            max_s,
                            worst_input.to_flat(),
                            worst_target.to_flat()
                        );
                    }
                } else {
                    println!(
                        "  [per-sample diag] suppressed (limit {} reached)",
                        ANOMALY_DIAG_LIMIT
                    );
                }
                println!();
            }
        }

        if log_epoch_summary_this_epoch {
            let min_loss = epoch_batches.iter().map(|b| b.loss).fold(f32::INFINITY, f32::min);
            let max_loss = epoch_batches.iter().map(|b| b.loss).fold(f32::NEG_INFINITY, f32::max);
            let (min_idx, _) = epoch_batches
                .iter()
                .enumerate()
                .min_by(|a, b| a.1.loss.partial_cmp(&b.1.loss).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or((0, &BatchInfo { start: 0, end: 0, loss: 0.0, grad_l2: 0.0 }));
            let (max_idx, _) = epoch_batches
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.loss.partial_cmp(&b.1.loss).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or((0, &BatchInfo { start: 0, end: 0, loss: 0.0, grad_l2: 0.0 }));
            println!(
                "  [ep-summary {}] avg={:.6} min={:.6}@batch[{}..{}] max={:.6}@batch[{}..{}] ratio={:.2}",
                epoch,
                avg_loss,
                min_loss,
                epoch_batches[min_idx].start,
                epoch_batches[min_idx].end,
                max_loss,
                epoch_batches[max_idx].start,
                epoch_batches[max_idx].end,
                if min_loss > 1e-9 { max_loss / min_loss } else { 0.0 },
            );

            if epoch < LOG_FIRST_EPOCHS || epoch % LOG_NORM_EVERY == 0 {
                if let Some(flat) = read_all_params_safe(model) {
                    train_dbg_print_stats(&format!("params @ epoch {}", epoch), &flat);
                }
            }
        }

        if avg_loss < best_loss {
            best_loss = avg_loss;
            best_epoch = epoch;
        }
        if avg_loss <= 0.0 && zero_loss_epoch.is_none() {
            zero_loss_epoch = Some(epoch);
        }

        if epoch % 10 == 0 || epoch == plan.epochs - 1 {
            println!("Epoch {}: avg loss = {:.6}", epoch, avg_loss);
        }

        if let Some(ref mut mon) = monitor {
            let summary = mon.end_epoch();
            if !summary.warnings.is_empty() {
                println!("--- Warnings after epoch {} ---", epoch);
                for w in &summary.warnings {
                    println!("  - {:?}", w);
                }
            }
        }

        if let Some(ref val_cfg) = plan.validation {
            if (epoch + 1) % val_cfg.frequency == 0 {
                let val_data = &val_cfg.data;
                let val_samples = val_data.num_samples();
                let mut val_loss = 0.0f32;
                for start in (0..val_samples).step_by(batch_size) {
                    let end = (start + batch_size).min(val_samples);
                    let batch = val_data.batch(start, end);
                    let (pred, _) = model.forward(batch.clone());
                    let (loss, _) = model.compute_loss(
                        plan.loss_desc.clone(),
                        &pred,
                        &batch,
                    );
                    val_loss += loss * (end - start) as f32;
                }
                if val_samples > 0 {
                    println!(
                        "Validation after epoch {}: avg loss = {:.6}",
                        epoch + 1,
                        val_loss / val_samples as f32
                    );
                }
            }
        }
    }

    let elapsed = start_time.elapsed().as_secs_f64();

    let mut result = TrainingResult {
        tensors: HashMap::new(),
        final_loss: 0.0,
        training_time_secs: elapsed,
        best_epoch,
        best_loss,
        zero_loss_epoch,
        profile: None,
        monitor_summary: None,
    };

    if let Some(test_input) = &plan.test_data {
        let test_input_dynamic = test_input.to_dynamic_tensor();
        let (pred, _) = model.forward(test_input_dynamic.clone());

        if plan.output_tensors.contains(&"prediction".to_string()) {
            result.tensors.insert("prediction".into(), pred.clone());
        }

        let target_for_test = match &plan.test_target_data {
            Some(t) => {
                assert_eq!(
                    test_input.num_samples(),
                    t.num_samples(),
                    "Test input and test target must have the same number of samples"
                );
                t.to_dynamic_tensor()
            }
            None => test_input_dynamic.clone(),
        };

        let (loss, _) = model.compute_loss(
            plan.loss_desc.clone(),
            &pred,
            &target_for_test,
        );
        result.final_loss = loss;
    }

    if plan.output_tensors.contains(&"loss".to_string()) {
        result.tensors.insert(
            "loss".into(),
            DynamicTensor::Dim1(Tensor2D::from_scalar(result.final_loss)),
        );
    }

    if let Some(prof) = profiler {
        result.profile = Some(prof.finish());
    }

    if let Some(mon) = monitor {
        let summary = mon.summary();
        println!("=== Training Monitor Summary ===");
        println!("Epochs: {}", summary.epochs);
        println!("Final loss: {:.6}", summary.final_loss);
        println!("NaN steps: {}", summary.nan_count);
        if !summary.warnings.is_empty() {
            println!("Warnings:");
            for w in &summary.warnings {
                println!("  - {:?}", w);
            }
        }
        result.monitor_summary = Some(summary);
    }

    Ok(result)
}