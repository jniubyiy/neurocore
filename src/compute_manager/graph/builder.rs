// src/compute_manager/graph/builder.rs

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::compute_manager::compute_executor::ComputeExecutor;
use crate::compute_manager::cpu::hardware::CPU_INFO;
use crate::compute_manager::cpu::{ComputeThreadPool, ControlThreadPool, CostModel, Scheduler};
use crate::compute_manager::device::Device;
use crate::compute_manager::executor::Executor;
use crate::compute_manager::graph::model::MixedModel;
use crate::compute_manager::matrix_buffer::TempMatrixPool;
use crate::compute_manager::memory_executor::types::MemoryDeviceKind;
use crate::device_plan::{ComputeDevice, DevicePlan};
use crate::layers::UniversalLayer;
use crate::model_plan::blueprint::LayerKind;
use crate::model_plan::layer_desc::LayerDesc;
use crate::model_plan::param_store::ParamStore;

use super::types::Model;

/// Автоматически разделяет доступные CPU-потоки на управляющие и вычислительные.
///
/// # Аргументы
/// * `total_threads` – общее количество CPU-потоков, указанное в `DevicePlan`.
/// * `has_gpu` – наличие GPU в плане устройств.
///
/// # Возвращает
/// Кортеж `(control_threads, compute_threads)`, где оба значения больше либо равны 1.
///
/// # Паника
/// Паникует, если `total_threads` меньше 2.
fn split_cpu_threads(total_threads: usize, has_gpu: bool) -> (usize, usize) {
    assert!(total_threads >= 2, "total_threads must be at least 2");
    let physical_cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(total_threads);

    // Базовое количество управляющих потоков: 1 на каждые 8 ядер, минимум 1, максимум 4.
    let mut control = ((physical_cores / 8).max(1)).min(4);

    // Не забираем все потоки под управление, оставляем минимум один вычислительный.
    control = control.min(total_threads - 1);

    let mut compute = total_threads - control;

    // Если есть GPU, можно сократить CPU-вычислительные потоки, но не до нуля.
    if has_gpu {
        compute = compute.min(physical_cores / 2);
    }

    // Гарантируем минимум один вычислительный поток.
    compute = compute.max(1);

    // На случай, если после предыдущих корректировок compute превысил доступное количество.
    compute = compute.min(total_threads - control);

    (control, compute)
}

impl MixedModel {
    pub(crate) fn from_plan(layers: Vec<LayerDesc>, num_threads: usize) -> Result<Self, String> {
        let plan = DevicePlan::empty()
            .cpu(0, num_threads)
            .ram(0, 8192);
        Self::from_plan_with_device_plan(layers, plan)
    }

    pub(crate) fn from_plan_with_device(
        layers: Vec<LayerDesc>,
        _num_threads: usize,
        device: Device,
    ) -> Result<Self, String> {
        let plan = match device {
            Device::Cpu { threads } => DevicePlan::empty().cpu(0, threads).ram(0, 8192),
            Device::Gpu { id } => DevicePlan::empty()
                .cpu(0, 2)
                .ram(0, 8192)
                .gpu(id)
                .vram(0, id, 4096),
        };
        Self::from_plan_with_device_plan_and_batch(layers, plan, 1)
    }

    /// Основной конструктор с планом устройств и размером батча.
    pub(crate) fn from_plan_with_device_plan_and_batch(
        layers: Vec<LayerDesc>,
        device_plan: DevicePlan,
        batch_size: usize,
    ) -> Result<Self, String> {
        // -----------------------------------------------------------
        // 1. Создаём MemoryExecutor
        // -----------------------------------------------------------
        let (memory_executor, _gpu_context) = device_plan.build_memory_executor();
        // Тип memory_executor: Arc<RwLock<MemoryExecutor>>

        // -----------------------------------------------------------
        // 2. Создаём вычислительный исполнитель (ComputeExecutor)
        // -----------------------------------------------------------
        let compute_executor = Arc::new(
            ComputeExecutor::new(device_plan.clone(), memory_executor.clone())
                .map_err(|e| format!("Failed to create ComputeExecutor: {}", e))?,
        );

        // -----------------------------------------------------------
        // 3. Определяем общее количество потоков CPU и наличие GPU
        // -----------------------------------------------------------
        let cpu_threads = compute_executor.cpu_threads();
        let has_gpu = device_plan
            .compute_devices
            .iter()
            .any(|d| matches!(d, ComputeDevice::Gpu { .. }));

        // -----------------------------------------------------------
        // 4. Разделяем потоки на управляющие и вычислительные
        // -----------------------------------------------------------
        let (control_threads, compute_threads) = split_cpu_threads(cpu_threads, has_gpu);

        // -----------------------------------------------------------
        // 5. Создаём пулы потоков
        // -----------------------------------------------------------
        let cost = CostModel::calibrate();
        let scheduler = Arc::new(Mutex::new(Scheduler::new_with_cpus(
            cost,
            CPU_INFO.clone(),
            compute_threads,
            has_gpu,
        )));
        scheduler.lock().unwrap().set_num_workers(compute_threads);

        let compute_executor_pool: Box<dyn Executor> =
            Box::new(ComputeThreadPool::new(compute_threads, scheduler.clone()));
        let control_executor_pool: Box<dyn Executor> =
            Box::new(ControlThreadPool::new(control_threads));

        // -----------------------------------------------------------
        // 6. Создаём хранилище параметров (ParamStore)
        // -----------------------------------------------------------
        let param_store = Arc::new(Mutex::new(ParamStore::new(memory_executor.clone())));

        // -----------------------------------------------------------
        // 7. Строим модели вычислительного графа
        // -----------------------------------------------------------
        let mut models: Vec<Model> = Vec::new();
        let mut current_layers: Vec<Box<dyn UniversalLayer>> = Vec::new();
        let mut current_layer_sizes: Vec<usize> = Vec::new();
        let mut active_ports: Option<Vec<usize>> = None;
        let mut current_branch: Option<usize> = None;
        let mut current_stream_indices: Option<Vec<usize>> = None;

        macro_rules! finalize_universal {
            () => {
                if !current_layers.is_empty() {
                    let slices = {
                        let mut ps = param_store.lock().unwrap();
                        ps.allocate_segment(&current_layer_sizes, MemoryDeviceKind::HostRam)
                    };
                    debug_assert_eq!(
                        slices.len(),
                        current_layers.len(),
                        "Number of slices must match number of layers"
                    );
                    models.push(Model::UniversalProcessor(
                        Arc::new(std::mem::take(&mut current_layers)),
                        slices,
                        current_stream_indices.take(),
                    ));
                    current_layer_sizes.clear();
                }
            };
        }

        for desc in &layers {
            match &desc.kind {
                LayerKind::SplitterConnector => {
                    finalize_universal!();

                    let dims: Vec<usize> = if !desc.output_shape.streams.is_empty() {
                        desc.output_shape.streams.clone()
                    } else {
                        desc.input_shape.streams.clone()
                    };

                    if dims.len() == 1 {
                        if let Some(ref ports) = active_ports {
                            if let Some(pos) = ports.iter().position(|&p| p == dims[0]) {
                                current_branch = Some(pos);
                            } else {
                                current_branch = Some(0);
                            }
                        } else {
                            active_ports = Some(vec![dims[0]]);
                            current_branch = Some(0);
                        }
                    } else {
                        active_ports = Some(dims.clone());
                        current_branch = None;
                    }
                }
                LayerKind::CombinerConnector => {
                    continue;
                }
                LayerKind::Splitter => {
                    finalize_universal!();
                    let input_dim = desc.input_shape.streams[0];
                    let output_dims = desc.output_shape.streams.clone();
                    active_ports = Some(output_dims.clone());

                    let slice = {
                        let mut ps = param_store.lock().unwrap();
                        let slices = ps.allocate_segment(&[desc.param_len()], MemoryDeviceKind::HostRam);
                        slices[0]
                    };
                    models.push(Model::Splitter {
                        input_dim,
                        output_dims,
                        slice,
                    });
                    current_branch = Some(0);
                }
                LayerKind::Combiner => {
                    finalize_universal!();
                    let input_dim = desc.input_shape.streams[0];
                    let output_dim = desc.output_shape.streams[0];

                    let slice = {
                        let mut ps = param_store.lock().unwrap();
                        let slices = ps.allocate_segment(&[desc.param_len()], MemoryDeviceKind::HostRam);
                        slices[0]
                    };
                    models.push(Model::Combiner {
                        input_dim,
                        output_dim,
                        slice,
                    });
                    active_ports = Some(vec![output_dim]);
                    current_branch = None;
                }
                LayerKind::Unsqueeze => {
                    finalize_universal!();
                    let target_dims = desc.output_shape.streams.clone();
                    models.push(Model::Unsqueeze(target_dims));
                }
                LayerKind::ReduceMean => {
                    finalize_universal!();
                    let target_dims = desc.output_shape.streams.clone();
                    models.push(Model::ReduceMean(target_dims));
                }
                _ => {
                    if current_stream_indices.is_none() {
                        let indices = if let Some(ref ports) = active_ports {
                            if let Some(ref mut branch) = current_branch {
                                if let Some(pos) = ports
                                    .iter()
                                    .position(|&p| p == desc.input_shape.streams[0])
                                {
                                    *branch = pos;
                                }
                            } else if let Some(pos) = ports
                                .iter()
                                .position(|&p| p == desc.input_shape.streams[0])
                            {
                                current_branch = Some(pos);
                            } else {
                                current_branch = Some(0);
                            }
                            Some(vec![current_branch.unwrap()])
                        } else {
                            None
                        };
                        current_stream_indices = indices;
                    }

                    let layer = desc.create_universal_layer();
                    current_layer_sizes.push(desc.param_len());
                    current_layers.push(layer);
                }
            }
        }
        finalize_universal!();

        let input_stream_count = match models.first() {
            Some(Model::CombinerConnector { input_dims, .. }) => input_dims.len(),
            _ => 1,
        };
        let output_stream_count = match models.last() {
            Some(Model::SplitterConnector { .. }) | Some(Model::Splitter { .. }) => 2,
            _ => 1,
        };

        // -----------------------------------------------------------
        // 8. Вычисляем ожидаемые формы входных и выходных тензоров
        // -----------------------------------------------------------
        let input_shapes: Vec<Vec<usize>> =
            vec![layers.first().unwrap().input_shape.streams.clone()];
        let output_shapes: Vec<Vec<usize>> = if output_stream_count == 1 {
            vec![layers.last().unwrap().output_shape.streams.clone()]
        } else {
            let last_model = models.last().unwrap();
            match last_model {
                Model::Splitter { output_dims, .. } => {
                    output_dims.iter().map(|&d| vec![d]).collect()
                }
                Model::SplitterConnector { dim_a, dim_b } => {
                    vec![vec![*dim_a], vec![*dim_b]]
                }
                _ => vec![],
            }
        };

        // -----------------------------------------------------------
        // 9. Создаём пул временных матриц
        // -----------------------------------------------------------
        let temp_matrix_pool = Arc::new(Mutex::new(TempMatrixPool::new(memory_executor.clone())));

        // -----------------------------------------------------------
        // 10. Выполняем начальное размещение моделей
        // -----------------------------------------------------------
        compute_executor.redistribute(&models, batch_size, true);

        // -----------------------------------------------------------
        // 11. Оборачиваем модели в Arc для совместного использования
        // -----------------------------------------------------------
        let models = Arc::new(models);

        // -----------------------------------------------------------
        // 12. Собираем MixedModel
        // -----------------------------------------------------------
        let model = MixedModel {
            models,
            param_store,
            executor: compute_executor_pool,
            control_executor: control_executor_pool,
            compute_executor,
            input_stream_count,
            output_stream_count,
            memory_executor,
            input_shapes,
            output_shapes,
            temp_matrix_pool,
            optimizer_exprs: HashMap::new(),
            last_forward_contexts: HashMap::new(),
        };

        Ok(model)
    }

    /// Обратно-совместимый конструктор (без batch_size, использует 1).
    pub(crate) fn from_plan_with_device_plan(
        layers: Vec<LayerDesc>,
        device_plan: DevicePlan,
    ) -> Result<Self, String> {
        Self::from_plan_with_device_plan_and_batch(layers, device_plan, 1)
    }

    /// Сборка модели с планом устройств (вызывается из публичного API).
    #[allow(dead_code)]
    pub(crate) fn build_with_device_plan(
        plan: crate::model_plan::plan::Plan,
        device_plan: DevicePlan,
    ) -> Result<Self, String> {
        Self::from_plan_with_device_plan_and_batch(plan.layers, device_plan, 1)
    }

    /// Сборка модели с планом устройств и указанием размера батча.
    #[allow(dead_code)]
    pub(crate) fn build_with_device_plan_and_batch(
        plan: crate::model_plan::plan::Plan,
        device_plan: DevicePlan,
        batch_size: usize,
    ) -> Result<Self, String> {
        Self::from_plan_with_device_plan_and_batch(plan.layers, device_plan, batch_size)
    }
}