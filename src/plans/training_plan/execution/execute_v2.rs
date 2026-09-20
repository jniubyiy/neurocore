// src/plans/training_plan/execution/execute_v2.rs
//
// Оркестратор обучения v2.
//
// # Фазы шага обучения (MIGRATION_PLAN.md §7, Фаза 4)
//
// Внутренний цикл обучения использует три явные фазы шага оптимизатора:
//
//   forward → loss → backward
//     → optimizer_modify_grads
//     → adapter_pass
//     → optimizer_apply_update
//
// Между `modify_grads` и `apply_update` встаёт `adapter_pass` — он обходит
// слои, у которых есть `GradientAdapter`, и вызывает `apply(&ctx)`.
//
// # Диагностика адаптеров (MIGRATION_PLAN.md §7, Фаза 6)
//
// `adapter_pass` возвращает `AdapterPassStats` — статистику работы
// адаптеров за шаг. `execute_inner_v2` аккумулирует её в `AdapterSummary`
// и по завершении обучения кладёт в `TrainingResult::adapter_summary`.
//
// L2-метрики (mean_scale и т. п.) заполняются только при
// `NEUROCORE_DEBUG_ADAPTER=1`. При `NEUROCORE_DISABLE_ADAPTERS=1`
// `adapter_pass` возвращает пустую статистику (baseline-режим).

use std::collections::HashMap;
use std::path::PathBuf;
use std::thread;
use std::time::Instant;

use once_cell::sync::Lazy;

use crate::compute_manager::core::dim_change::DynamicTensor;
use crate::compute_manager::graph_v2::{bridge_v2, GraphV2};
use crate::compute_manager::operators_v2::memory_v2::types::MemoryDeviceKind;
use crate::device_plan::DevicePlan;
use crate::layers::adapter::AdapterSummary;
use crate::logging::training_monitor::TrainingMonitor;
use crate::tensor::Tensor2D;
use crate::training_plan::plan::TrainingPlan;
use crate::training_plan::profiling::{ProfileMode, Profiler};

use super::types::TrainingResult;

// ============================================================================
// Env-флаги диагностики (MIGRATION_PLAN.md §7, Фаза 6)
// ============================================================================

/// `NEUROCORE_DEBUG_ADAPTER=1` — печатать сводку адаптеров после обучения.
///
/// Сама диагностика (замер L2, лог `[ADAPTER ...]`) управляется внутри
/// `GraphV2::adapter_pass` (см. `graph_v2::model_v2`). Здесь флаг нужен
/// только для того, чтобы не сорить в консоль `AdapterSummary::report()`
/// без запроса.
static DEBUG_ADAPTER: Lazy<bool> =
    Lazy::new(|| std::env::var("NEUROCORE_DEBUG_ADAPTER").is_ok());

// ============================================================================
// Публичный API
// ============================================================================

/// Запускает обучение v2 в отдельном потоке с увеличенным стеком.
pub fn execute(
    plan: &TrainingPlan,
    device_plan: &DevicePlan,
) -> Result<TrainingResult, String> {
    let plan = plan.clone();
    let device_plan = device_plan.clone();

    let handle = thread::Builder::new()
        .name("neurocore-v2-main".to_string())
        .stack_size(32 * 1024 * 1024)
        .spawn(move || {
            let model_desc = (plan.model_fn)();
            let mut graph = GraphV2::build(model_desc, &device_plan)?;
            if let Some(seed) = plan.seed {
                graph = graph.with_seed(seed);
            }
            execute_inner_v2(&plan, &mut graph)
        })
        .map_err(|e| format!("execute_v2: failed to spawn training thread: {}", e))?;

    match handle.join() {
        Ok(inner) => inner,
        Err(_) => Err("execute_v2: training thread panicked".to_string()),
    }
}

// ============================================================================
// Внутренний цикл
// ============================================================================

fn execute_inner_v2(
    plan: &TrainingPlan,
    graph: &mut GraphV2,
) -> Result<TrainingResult, String> {
    let start_time = Instant::now();

    // ---------------------------------------------------------------------
    // 0. Проверка многопоточных DataSource
    // ---------------------------------------------------------------------
    if plan.train_data_streams.is_some()
        || plan.target_data_streams.is_some()
        || plan.test_input_streams.is_some()
        || plan.test_target_streams.is_some()
    {
        return Err(
            "Многопоточковые данные (train_data_streams, target_data_streams, \
             test_input_streams, test_target_streams) пока не поддерживаются \
             в автоматическом обучении v2. Используйте ручной цикл с \
             forward_multi/backward_multi."
                .to_string(),
        );
    }

    // ---------------------------------------------------------------------
    // 1. Инициализация параметров
    // ---------------------------------------------------------------------
    graph.init_params(plan.initializer.clone())?;
    apply_layer_aware_overrides_v2(graph)?;

    // ---------------------------------------------------------------------
    // 2. Создание монитора и профайлера
    // ---------------------------------------------------------------------
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

    let mut profiler = if plan.profile != ProfileMode::None {
        Some(Profiler::new(plan.profile))
    } else {
        None
    };

    // Аккумулятор статистики адаптеров (Фаза 6 плана).
    // Заполняется после каждого adapter_pass.
    let mut adapter_summary_acc = AdapterSummary::default();

    // ---------------------------------------------------------------------
    // 3. Подготовка данных
    // ---------------------------------------------------------------------
    let train_data = plan
        .train_data
        .clone()
        .ok_or_else(|| "execute_v2: train_data not provided".to_string())?;
    let target_data = plan
        .target_data
        .clone()
        .unwrap_or_else(|| train_data.clone());

    if train_data.num_samples() != target_data.num_samples() {
        return Err(format!(
            "execute_v2: train/target sample count mismatch ({} vs {})",
            train_data.num_samples(),
            target_data.num_samples()
        ));
    }

    let num_samples = train_data.num_samples();
    let batch_size = plan.batch_size.max(1);

    // ---------------------------------------------------------------------
    // 4. Цикл эпох
    // ---------------------------------------------------------------------
    let mut best_loss = f32::MAX;
    let mut best_epoch: usize = 0;
    let mut zero_loss_epoch: Option<usize> = None;

    for epoch in 0..plan.epochs {
        graph.observe_epoch(epoch)?;

        let mut epoch_loss_sum = 0.0f32;
        let mut epoch_samples = 0usize;

        for start in (0..num_samples).step_by(batch_size) {
            let end = (start + batch_size).min(num_samples);
            if end <= start {
                continue;
            }
            let actual = end - start;

            let x = train_data.batch(start, end);
            let y = target_data.batch(start, end);

            // --- Профилирование: forward ---
            let t0 = Instant::now();
            let pred = graph.forward(x)?;
            let forward_dt = t0.elapsed().as_nanos() as u64;

            // --- Профилирование: loss ---
            let t1 = Instant::now();
            let (loss, delta) = graph.loss(plan.loss_desc.clone(), &pred, &y)?;
            let loss_dt = t1.elapsed().as_nanos() as u64;

            // --- Профилирование: backward ---
            let t2 = Instant::now();
            let _grad_input = graph.backward(delta)?;
            let backward_dt = t2.elapsed().as_nanos() as u64;

            // --- Сбор градиентов ДО optimizer_modify_grads (как в v1) ---
            let grads_flat: Option<Vec<f32>> = if monitor.is_some() {
                collect_grads_flat(graph)
            } else {
                None
            };

            // --- Три фазы шага оптимизатора ---
            let t3 = Instant::now();

            graph.optimizer_modify_grads(plan.optimizer_desc.clone())?;
            let modify_dt = t3.elapsed().as_nanos() as u64;

            let t_ad = Instant::now();
            let adapter_stats = graph.adapter_pass()?;
            let adapter_dt = t_ad.elapsed().as_nanos() as u64;

            // Аккумулируем статистику адаптеров (Фаза 6 плана).
            adapter_summary_acc.absorb(&adapter_stats);

            let t_ap = Instant::now();
            graph.optimizer_apply_update(plan.optimizer_desc.clone())?;
            let apply_dt = t_ap.elapsed().as_nanos() as u64;

            let update_dt = modify_dt + adapter_dt + apply_dt;

            graph.observe_step(loss, None);

            if let Some(ref mut prof) = profiler {
                if prof.mode == ProfileMode::Time || prof.mode == ProfileMode::Full {
                    prof.record_timing(0, "batch", "V2", "forward", forward_dt);
                    prof.record_timing(0, "batch", "V2", "loss", loss_dt);
                    prof.record_timing(0, "batch", "V2", "backward", backward_dt);
                    prof.record_timing(0, "batch", "V2", "update", update_dt);
                }
                if prof.mode == ProfileMode::Memory || prof.mode == ProfileMode::Full {
                    let snapshot = graph.distributor().snapshot();
                    let me = snapshot.memory_executor.read().unwrap();
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

            if let Some(ref mut mon) = monitor {
                mon.record_step(loss, grads_flat.as_deref(), None);
            }

            epoch_loss_sum += loss * actual as f32;
            epoch_samples += actual;
        }

        let avg_loss = if epoch_samples > 0 {
            epoch_loss_sum / epoch_samples as f32
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

        let report = graph.end_epoch();

        if !report.warnings.is_empty() {
            println!(
                "Epoch {}: avg_loss = {:.6}  ({} warnings)",
                epoch,
                avg_loss,
                report.warnings.len()
            );
            for w in &report.warnings {
                println!("  {:?}", w);
            }
            for rec in &report.recommendations {
                println!("  → {}", rec);
            }
        } else if epoch % 10 == 0 || epoch == plan.epochs - 1 {
            println!("Epoch {}: avg_loss = {:.6}", epoch, avg_loss);
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

        // --- Валидация ---
        if let Some(ref val_cfg) = plan.validation {
            if (epoch + 1) % val_cfg.frequency == 0 {
                let val_data = &val_cfg.data;
                let val_samples = val_data.num_samples();
                let mut val_loss = 0.0f32;
                let mut val_seen = 0usize;

                for start in (0..val_samples).step_by(batch_size) {
                    let end = (start + batch_size).min(val_samples);
                    if end <= start {
                        continue;
                    }
                    let val_batch = val_data.batch(start, end);
                    let pred = graph.forward(val_batch.clone())?;
                    let (loss, _) = graph.loss(
                        plan.loss_desc.clone(),
                        &pred,
                        &val_batch,
                    )?;
                    val_loss += loss * (end - start) as f32;
                    val_seen += end - start;
                }

                if val_seen > 0 {
                    println!(
                        "Validation after epoch {}: avg loss = {:.6}",
                        epoch + 1,
                        val_loss / val_seen as f32
                    );
                }
            }
        }
    }

    // ---------------------------------------------------------------------
    // 5. Тестовый прогон
    // ---------------------------------------------------------------------
    let mut final_loss = 0.0f32;
    let mut tensors: HashMap<String, DynamicTensor> = HashMap::new();

    if let Some(test_input) = &plan.test_data {
        let test_input_dynamic = test_input.to_dynamic_tensor();
        let pred = graph.forward(test_input_dynamic.clone())?;

        if plan.output_tensors.contains(&"prediction".to_string()) {
            tensors.insert("prediction".to_string(), pred.clone());
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

        let (loss, _) = graph.loss(plan.loss_desc.clone(), &pred, &target_for_test)?;
        final_loss = loss;
    }

    let elapsed = start_time.elapsed().as_secs_f64();

    let mut result = TrainingResult {
        tensors,
        final_loss,
        training_time_secs: elapsed,
        best_epoch,
        best_loss,
        zero_loss_epoch,
        profile: None,
        monitor_summary: None,
        adapter_summary: None,
    };

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

    // ---------------------------------------------------------------------
    // 6. Сводка адаптеров (Фаза 6 плана)
    // ---------------------------------------------------------------------
    //
    // `Some`, если за обучение был хотя бы один вызов адаптера.
    // При `NEUROCORE_DISABLE_ADAPTERS=1` — всегда `None` (adapter_pass
    // возвращал пустые stats).
    if adapter_summary_acc.total_calls > 0 {
        if *DEBUG_ADAPTER {
            println!("=== Adapter Summary ===");
            print!("{}", adapter_summary_acc.report());
        }
        result.adapter_summary = Some(adapter_summary_acc);
    }

    Ok(result)
}

// ============================================================================
// Применение канонических per-layer инициализаций
// ============================================================================

/// Применяет `bridge_v2::build_layer_aware_overrides_v2` поверх
/// generic-инициализации.
fn apply_layer_aware_overrides_v2(graph: &GraphV2) -> Result<(), String> {
    let overrides = bridge_v2::build_layer_aware_overrides_v2(graph);
    if overrides.is_empty() {
        return Ok(());
    }

    let ps_arc = graph.param_store().clone();
    let ps = ps_arc
        .lock()
        .map_err(|_| "apply_layer_aware_overrides_v2: param_store poisoned".to_string())?;

    for (buffer_idx, start, buf) in overrides {
        let buffer = ps.get_param_buffer_by_idx(buffer_idx);
        buffer.params.write_range(start, &buf);
    }

    Ok(())
}

// ============================================================================
// Вспомогательные функции
// ============================================================================

/// Собирает плоский вектор градиентов из всех буферов `ParamStore`.
fn collect_grads_flat(graph: &GraphV2) -> Option<Vec<f32>> {
    let ps = graph.param_store().lock().ok()?;
    let total = ps.total_params();
    if total == 0 {
        return Some(Vec::new());
    }

    let gpu_opt = graph.distributor().gpu_compute();

    let mut out: Vec<f32> = Vec::with_capacity(total);
    for i in 0..ps.num_buffers() {
        let b = ps.get_param_buffer_by_idx(i);
        if b.grads.is_gpu() {
            let gpu = gpu_opt.as_ref()?;
            let vec = gpu.download_gpu_handle_to_vec(&b.grads);
            out.extend_from_slice(&vec);
        } else {
            let guard = b.grads.read();
            let slice = guard.as_slice().expect("CPU buffer");
            out.extend_from_slice(slice);
        }
    }
    Some(out)
}