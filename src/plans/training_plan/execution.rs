// src/plans/training_plan/execution.rs

use std::collections::HashMap;
use std::path::PathBuf;
use std::thread;
use std::time::Instant;

use rand::Rng;
use rand::SeedableRng;

use crate::compute_manager::dim_change::DynamicTensor;
use crate::compute_manager::graph::model::MixedModel;
use crate::device_plan::{ComputeDevice, DevicePlan};
use crate::logging::training_monitor::TrainingMonitor;
use crate::model_plan::Plan;
use crate::tensor::Tensor2D;

use super::plan::{DataSource, Initializer, TrainingPlan};
use super::profiling::{Profiler, ProfileMode, ProfileResult};
use crate::compute_manager::memory_executor::types::MemoryDeviceKind;

/// Результат обучения с метриками и опциональным профилем.
pub struct TrainingResult {
    pub tensors: HashMap<String, DynamicTensor>,
    pub final_loss: f32,
    pub training_time_secs: f64,
    pub best_epoch: usize,
    pub best_loss: f32,
    pub zero_loss_epoch: Option<usize>,
    pub profile: Option<ProfileResult>,
    /// Сводка мониторинга обучения, если был включён.
    pub monitor_summary: Option<crate::logging::TrainingSummary>,
}

/// Выполняет план обучения. Для GPU запускает весь цикл в отдельном потоке
/// с увеличенным стеком, чтобы избежать переполнения в Vulkan-драйверах.
pub fn execute(plan: &TrainingPlan, device_plan: &DevicePlan) -> Result<TrainingResult, String> {
    let has_gpu = device_plan
        .compute_devices
        .iter()
        .any(|d| matches!(d, ComputeDevice::Gpu { .. }));

    if has_gpu {
        let plan = plan.clone();
        let device_plan = device_plan.clone();
        let handle = thread::Builder::new()
            .stack_size(32 * 1024 * 1024)
            .spawn(move || execute_inner(&plan, &device_plan))
            .map_err(|e| format!("Failed to spawn GPU training thread: {}", e))?;
        handle
            .join()
            .map_err(|_| "GPU training thread panicked".to_string())?
    } else {
        execute_inner(plan, device_plan)
    }
}

fn execute_inner(plan: &TrainingPlan, device_plan: &DevicePlan) -> Result<TrainingResult, String> {
    let start_time = Instant::now();

    // --- построение модели ---
    let model_desc = (plan.model_fn)();
    let _ = Plan::from_layer_descs(model_desc.clone())?;
    let model = MixedModel::from_plan_with_device_plan(model_desc, device_plan.clone())?;

    // --- инициализация весов ---
    {
        let mut store = model.param_store().lock().unwrap();
        let len = store.len();
        match &plan.initializer {
            Initializer::Zeros => store.set_all_params(&vec![0.0f32; len]),
            Initializer::Ones => store.set_all_params(&vec![1.0f32; len]),
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
                store.set_all_params(&params);
            }
        }
    }

    // --- оптимизатор (сначала строим цепочку, чтобы извлечь learning rate) ---
    let opt_chain = plan.optimizer_desc.build_chain();

    // --- learning rate для монитора (ищем первый ScaleGradient в цепочке) ---
    let learning_rate = opt_chain
        .cubes()
        .iter()
        .find_map(|cube| {
            cube.as_any()
                .downcast_ref::<crate::optimizer_plan::cubes::ScaleGradient>()
                .map(|sg| sg.factor)
        })
        .unwrap_or(0.01);

    let mut optimizer = model.create_optimizer(opt_chain);

    // --- монитор обучения ---
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

    // --- извлечение данных ---
    let train_data = match &plan.train_data {
        Some(data) => data.clone(),
        None => return Err("Training data not provided".into()),
    };
    let target_data = match &plan.target_data {
        Some(data) => data.clone(),
        None => train_data.clone(), // цель совпадает с входными данными (автоэнкодер)
    };
    assert_eq!(
        train_data.num_samples(),
        target_data.num_samples(),
        "Training data and target data must have the same number of samples"
    );

    let num_samples = train_data.num_samples();
    let batch_size = plan.batch_size.max(1);

    // --- профилировщик ---
    let mut profiler = if plan.profile != ProfileMode::None {
        Some(Profiler::new(plan.profile))
    } else {
        None
    };

    let mut best_loss = f32::MAX;
    let mut best_epoch = 0usize;
    let mut zero_loss_epoch: Option<usize> = None;

    for epoch in 0..plan.epochs {
        let mut epoch_loss = 0.0f32;

        for start in (0..num_samples).step_by(batch_size) {
            let end = (start + batch_size).min(num_samples);
            let batch_size_actual = end - start;

            // входной батч
            let batch_tensor = train_data.batch(start, end);
            // целевой батч
            let target_batch = target_data.batch(start, end);

            // forward
            let t0 = Instant::now();
            let (pred, ctxs) = model.forward(batch_tensor.clone());
            let forward_dt = t0.elapsed().as_nanos() as u64;

            // loss
            let t1 = Instant::now();
            let (loss, delta) = model.compute_loss(
                plan.loss_desc.clone(),
                &pred,
                &target_batch,
            );
            let loss_dt = t1.elapsed().as_nanos() as u64;

            // backward
            let t2 = Instant::now();
            let (_, grads) = model.backward(&ctxs, delta);
            let backward_dt = t2.elapsed().as_nanos() as u64;

            // обновление параметров
            let t3 = Instant::now();
            {
                let mut store = model.param_store().lock().unwrap();
                let mut params = store.all_params_vec();
                optimizer.step(&mut params, &grads[0]);
                store.set_all_params(&params);
            }
            let update_dt = t3.elapsed().as_nanos() as u64;

            epoch_loss += loss * batch_size_actual as f32;

            // --- мониторинг шага ---
            if let Some(ref mut mon) = monitor {
                mon.record_step(loss, Some(&grads[0]), None);
            }

            // профилирование одного шага
            if let Some(ref mut prof) = profiler {
                if prof.mode == ProfileMode::Time || prof.mode == ProfileMode::Full {
                    prof.record_timing(0, "batch", "CPU", "forward", forward_dt);
                    prof.record_timing(0, "batch", "CPU", "loss", loss_dt);
                    prof.record_timing(0, "batch", "CPU", "backward", backward_dt);
                    prof.record_timing(0, "batch", "CPU", "update", update_dt);
                }
                if prof.mode == ProfileMode::Memory || prof.mode == ProfileMode::Full {
                    let mem = model.memory_executor();
                    let me = mem.lock().unwrap();
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

        // --- завершение эпохи для монитора ---
        if let Some(ref mut mon) = monitor {
            let summary = mon.end_epoch();
            if !summary.warnings.is_empty() {
                println!("--- Warnings after epoch {} ---", epoch);
                for w in &summary.warnings {
                    println!("  - {:?}", w);
                }
            }
        }

        // --- валидация ---
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

    // --- финальный тест ---
    if let Some(test_data) = &plan.test_data {
        let test_dynamic = test_data.to_dynamic_tensor();
        let (pred, _) = model.forward(test_dynamic.clone());
        if plan.output_tensors.contains(&"prediction".to_string()) {
            result.tensors.insert("prediction".into(), pred);
        }
        let (loss, _) = model.compute_loss(
            plan.loss_desc.clone(),
            &result.tensors.get("prediction").unwrap(),
            &test_dynamic,
        );
        result.final_loss = loss;
    }

    if plan.output_tensors.contains(&"loss".to_string()) {
        result.tensors.insert(
            "loss".into(),
            DynamicTensor::Dim1(Tensor2D::from_scalar(result.final_loss)),
        );
    }

    // --- финализация профиля ---
    if let Some(prof) = profiler {
        result.profile = Some(prof.finish());
    }

    // --- итог мониторинга ---
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