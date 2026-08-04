// src/plans/training_plan/execution.rs

use std::collections::HashMap;
use std::thread;
use std::time::Instant;

use rand::Rng;
use rand::SeedableRng;

use crate::compute_manager::dim_change::DynamicTensor;
use crate::compute_manager::graph::model::MixedModel;
use crate::device_plan::{ComputeDevice, DevicePlan};
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

    // --- оптимизатор ---
    let opt_chain = plan.optimizer_desc.build_chain();
    let mut optimizer = model.create_optimizer(opt_chain);

    let train_data = match &plan.train_data {
        Some(DataSource::Tensor2D(t)) => t.clone(),
        None => return Err("Training data not provided".into()),
    };
    let num_samples = train_data.dim1;
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
            let rows: Vec<Vec<f32>> = (start..end)
                .map(|i| train_data.data[i].clone())
                .collect();
            let batch_tensor = Tensor2D::new(rows);

            // forward
            let t0 = Instant::now();
            let (pred, ctxs) = model.forward(DynamicTensor::Dim1(batch_tensor.clone()));
            let forward_dt = t0.elapsed().as_nanos() as u64;

            // loss
            let t1 = Instant::now();
            let (loss, delta) = model.compute_loss(
                plan.loss_desc.clone(),
                &pred,
                &DynamicTensor::Dim1(batch_tensor.clone()),
            );
            let loss_dt = t1.elapsed().as_nanos() as u64;

            // backward
            let t2 = Instant::now();
            let (_, grads) = model.backward(&ctxs, delta);
            let backward_dt = t2.elapsed().as_nanos() as u64;

            // update
            let t3 = Instant::now();
            {
                let mut store = model.param_store().lock().unwrap();
                let mut params = store.all_params_vec();
                optimizer.step(&mut params, &grads[0]);
                store.set_all_params(&params);
            }
            let update_dt = t3.elapsed().as_nanos() as u64;

            epoch_loss += loss * (end - start) as f32;

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
                    // Собираем использование по всем зарегистрированным типам памяти
                    let kinds = [
                        MemoryDeviceKind::HostRam,
                        // DeviceVram для каждого GPU — упрощённо берём все возможные (можно перебрать device_plan)
                        // для полноты добавим только HostRam и SsdCache
                        MemoryDeviceKind::SsdCache,
                    ];
                    for kind in &kinds {
                        let used = me.current_usage(*kind);
                        prof.record_memory(0, &format!("{:?}", kind), "batch", used, used);
                    }
                    // Для DeviceVram нужно перебирать GPU, но пока пропустим, чтобы не усложнять
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

        // --- валидация ---
        if let Some(ref val_cfg) = plan.validation {
            if (epoch + 1) % val_cfg.frequency == 0 {
                let DataSource::Tensor2D(val_data) = &val_cfg.data;
                let val_samples = val_data.dim1;
                let mut val_loss = 0.0f32;
                for start in (0..val_samples).step_by(batch_size) {
                    let end = (start + batch_size).min(val_samples);
                    let rows: Vec<Vec<f32>> = (start..end)
                        .map(|i| val_data.data[i].clone())
                        .collect();
                    let batch = Tensor2D::new(rows);
                    let (pred, _) = model.forward(DynamicTensor::Dim1(batch.clone()));
                    let (loss, _) = model.compute_loss(
                        plan.loss_desc.clone(),
                        &pred,
                        &DynamicTensor::Dim1(batch.clone()),
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
    };

    // --- финальный тест ---
    if let Some(test_data) = &plan.test_data {
        let DataSource::Tensor2D(t) = test_data;
        let (pred, _) = model.forward(DynamicTensor::Dim1(t.clone()));
        if plan.output_tensors.contains(&"prediction".to_string()) {
            result.tensors.insert("prediction".into(), pred);
        }
        let (loss, _) = model.compute_loss(
            plan.loss_desc.clone(),
            &result.tensors.get("prediction").unwrap(),
            &DynamicTensor::Dim1(t.clone()),
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

    Ok(result)
}