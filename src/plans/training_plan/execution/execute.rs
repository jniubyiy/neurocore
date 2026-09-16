// src/plans/training_plan/execution/execute.rs
//
// Оркестраторы обучения: публичный `execute` и внутренний `execute_inner`.

use std::collections::HashMap;
use std::path::PathBuf;
use std::thread;
use std::time::Instant;

use rand::Rng;
use rand::SeedableRng;

use crate::compute_manager::dim_change::DynamicTensor;
use crate::compute_manager::graph::model::MixedModel;
use crate::compute_manager::memory_executor::types::MemoryDeviceKind;
use crate::device_plan::DevicePlan;
use crate::logging::training_monitor::TrainingMonitor;
use crate::model_plan::Plan;
use crate::tensor::Tensor2D;

use super::super::plan::{Initializer, TrainingPlan};
use super::super::profiling::{Profiler, ProfileMode};

use super::debug::{
    TRAIN_DEBUG, LAYER_DEBUG, NAN_DEBUG,
    train_dbg_l2, train_dbg_print_vec, train_dbg_print_stats, train_dbg_summary,
    LOG_FIRST_EPOCHS, LOG_NORM_EVERY,
    ANOMALY_MULTIPLIER, ANOMALY_MIN_LOSS, ANOMALY_DIAG_LIMIT,
    LOG_LSF_EVO_FIRST, LOG_LSF_EVO_EVERY,
};
use super::lsf_evo::{collect_lsf_evo, print_lsf_evo};
use super::nan_debug::{contains_bad, report_first_nan, NanTracker};
use super::overrides::build_layer_aware_overrides;
use super::params::{per_sample_mse, read_all_params_safe};
use super::types::{BatchInfo, TrainingResult};

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
            "Многопоточковые данные (train_data_streams, target_data_streams, \
             test_input_streams, test_target_streams) пока не поддерживаются \
             в автоматическом обучении. Используйте ручной цикл с forward_multi/backward_multi."
                .to_string(),
        );
    }

    // --- Инициализация весов ---
    //
    // 1. Применяем пользовательский Initializer ко ВСЕМ параметрам.
    // 2. Применяем layer-aware канонические инициализации для BN-подобных
    //    слоёв (перезаписывая generic-init).
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

        // Layer-aware overrides.
        let overrides = build_layer_aware_overrides(model);
        for (buffer_idx, start, buf) in overrides {
            let handle = ps.get_param_buffer_by_idx(buffer_idx).params.clone();
            // write_range на CPU-буфере выполняется напрямую (без выделения
            // временной копии); на GPU-буфере запаникует — но на этапе
            // инициализации параметры всегда на CPU (см. set_all_params).
            handle.write_range(start, &buf);
        }
    }

    // --- Проверка инициализации на NaN ---
    let mut nan_tracker = NanTracker::new();
    if *NAN_DEBUG {
        if let Some(flat) = read_all_params_safe(model) {
            if contains_bad(&flat) && nan_tracker.should_report("init") {
                report_first_nan(
                    "init",
                    0, 0,
                    &[("params_after_init", flat.len(), 1)],
                    &flat,
                    Some("NaN/Inf обнаружен в параметрах СРАЗУ после инициализации"),
                );
                nan_tracker.mark_reported("init");
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

            nan_tracker.bump();

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

            if *NAN_DEBUG && nan_tracker.should_report("forward") {
                let pred_flat = pred.to_flat();
                if contains_bad(&pred_flat) {
                    report_first_nan(
                        "forward",
                        epoch,
                        batch_idx,
                        &[("pred (flat)", 1, pred_flat.len()),
                          ("batch_tensor (flat)", 1, batch_tensor.to_flat().len())],
                        &pred_flat,
                        Some(&format!(
                            "epoch={} batch_idx={} range=[{}..{})",
                            epoch, batch_idx, start, end
                        )),
                    );
                    nan_tracker.mark_reported("forward");
                }
            }

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

            if *NAN_DEBUG && nan_tracker.should_report("loss") {
                let delta_flat = delta.to_flat();
                let pred_flat = pred.to_flat();
                let target_flat = target_batch.to_flat();
                let bad = loss.is_nan() || loss.is_infinite() || contains_bad(&delta_flat);
                if bad {
                    let shapes = [
                        ("pred", 1, pred_flat.len()),
                        ("target", 1, target_flat.len()),
                        ("delta", 1, delta_flat.len()),
                    ];
                    report_first_nan(
                        "loss",
                        epoch,
                        batch_idx,
                        &shapes,
                        &delta_flat,
                        Some(&format!(
                            "epoch={} batch_idx={} loss={:?} pred_summary=[{}] target_summary=[{}]",
                            epoch, batch_idx, loss,
                            train_dbg_summary(&pred_flat),
                            train_dbg_summary(&target_flat),
                        )),
                    );
                    nan_tracker.mark_reported("loss");
                }
            }

            let t2 = Instant::now();
            let (_, grads) = model.backward(delta);
            let backward_dt = t2.elapsed().as_nanos() as u64;

            let grads_flat = grads.to_flat_vec();
            let grad_l2 = train_dbg_l2(&grads_flat);

            if *NAN_DEBUG && nan_tracker.should_report("backward") {
                if contains_bad(&grads_flat) {
                    report_first_nan(
                        "backward",
                        epoch,
                        batch_idx,
                        &[("grads (flat)", 1, grads_flat.len())],
                        &grads_flat,
                        Some(&format!(
                            "epoch={} batch_idx={} range=[{}..{})",
                            epoch, batch_idx, start, end
                        )),
                    );
                    nan_tracker.mark_reported("backward");
                }
            }

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

            if *NAN_DEBUG && nan_tracker.should_report("update") {
                if let Some(flat) = read_all_params_safe(model) {
                    if contains_bad(&flat) {
                        report_first_nan(
                            "update",
                            epoch,
                            batch_idx,
                            &[("params_after_update (flat)", 1, flat.len())],
                            &flat,
                            Some(&format!(
                                "epoch={} batch_idx={} range=[{}..{})",
                                epoch, batch_idx, start, end
                            )),
                        );
                        nan_tracker.mark_reported("update");
                    }
                }
            }

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

        if *TRAIN_DEBUG {
            let log_evo = epoch < LOG_LSF_EVO_FIRST
                || epoch % LOG_LSF_EVO_EVERY == 0
                || epoch == plan.epochs - 1;
            if log_evo {
                let snaps = collect_lsf_evo(model);
                print_lsf_evo(epoch, &snaps);
            }
        }

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

    if *NAN_DEBUG {
        println!();
        println!("[NAN-DEBUG] observed steps: {}", nan_tracker.total_steps_observed);
        println!("[NAN-DEBUG] reported stages:");
        println!("  init     : {}", nan_tracker.should_report("init") == false);
        println!("  forward  : {}", nan_tracker.should_report("forward") == false);
        println!("  loss     : {}", nan_tracker.should_report("loss") == false);
        println!("  backward : {}", nan_tracker.should_report("backward") == false);
        println!("  update   : {}", nan_tracker.should_report("update") == false);
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