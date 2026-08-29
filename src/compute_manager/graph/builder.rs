// src/compute_manager/graph/builder.rs

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::compute_manager::compute_executor::ComputeExecutor;
use crate::compute_manager::cpu::{CostModel, Scheduler, WorkerPool};
use crate::compute_manager::cpu::hardware::CPU_INFO;
use crate::compute_manager::device::Device;
use crate::compute_manager::executor::Executor;
use crate::compute_manager::graph::model::MixedModel;
use crate::compute_manager::matrix_buffer::TempMatrixPool;
use crate::device_plan::{ComputeDevice, DevicePlan};
use crate::layers::UniversalLayer;
use crate::model_plan::blueprint::LayerKind;
use crate::model_plan::layer_desc::LayerDesc;
use crate::model_plan::param_store::{ParamSlice, ParamStore};
use crate::compute_manager::memory_executor::types::MemoryDeviceKind;

use super::types::Segment;

// ---------- CpuExecutor ----------
#[derive(Clone)]
struct CpuExecutor {
    pool: Arc<WorkerPool>,
    scheduler: Arc<Mutex<Scheduler>>,
}

impl CpuExecutor {
    fn new(pool: Arc<WorkerPool>, scheduler: Arc<Mutex<Scheduler>>) -> Self {
        Self { pool, scheduler }
    }
}

impl Executor for CpuExecutor {
    fn execute_dyn(&self, f: Box<dyn FnOnce() + Send>) {
        self.pool.execute(f);
    }
    fn wait_all(&self) {
        self.pool.wait_all();
    }
    fn num_workers(&self) -> usize {
        self.scheduler.lock().unwrap().num_workers()
    }
    fn plan_chunks_assignment(&self, total_tasks: usize) -> Vec<Vec<(usize, usize)>> {
        self.scheduler.lock().unwrap().plan_chunks_assignment(total_tasks)
    }
    fn clone_executor(&self) -> Box<dyn Executor> {
        Box::new(self.clone())
    }
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
            Device::Gpu { id } => DevicePlan::empty().cpu(0, 2).ram(0, 8192).gpu(id).vram(0, id, 4096),
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

        // -----------------------------------------------------------
        // 2. Создаём вычислительный исполнитель (ComputeExecutor)
        // -----------------------------------------------------------
        let compute_executor = Arc::new(
            ComputeExecutor::new(device_plan.clone(), memory_executor.clone())
                .map_err(|e| format!("Failed to create ComputeExecutor: {}", e))?
        );

        // -----------------------------------------------------------
        // 3. Суммарное количество потоков CPU
        // -----------------------------------------------------------
        let cpu_threads = compute_executor.cpu_threads();
        let num_cpus = device_plan.compute_devices.iter()
            .filter(|d| matches!(d, ComputeDevice::Cpu { .. }))
            .count()
            .max(1);

        // -----------------------------------------------------------
        // 4. Создаём хранилище параметров (ParamStore)
        // -----------------------------------------------------------
        let param_store = Arc::new(Mutex::new(ParamStore::new(memory_executor.clone())));

        // -----------------------------------------------------------
        // 5. Создаём CPU-исполнитель (WorkerPool + Scheduler)
        // -----------------------------------------------------------
        let cost = CostModel::calibrate();
        let mut scheduler = Scheduler::new_with_cpus(cost, CPU_INFO.clone(), num_cpus);
        scheduler.set_num_workers(cpu_threads);
        let pool = Arc::new(WorkerPool::new(cpu_threads));
        let cpu_executor: Box<dyn Executor> = Box::new(
            CpuExecutor::new(pool.clone(), Arc::new(Mutex::new(scheduler.clone())))
        );

        // -----------------------------------------------------------
        // 6. Строим сегменты модели
        // -----------------------------------------------------------
        let mut segments: Vec<Segment> = Vec::new();
        let mut current_layers: Vec<Box<dyn UniversalLayer>> = Vec::new();
        let mut current_layer_sizes: Vec<usize> = Vec::new();
        let mut active_ports: Option<Vec<usize>> = None;
        let mut current_branch: Option<usize> = None;
        let mut current_stream_indices: Option<Vec<usize>> = None;

        // Макрос для финализации текущего UniversalProcessor сегмента.
        macro_rules! finalize_universal {
            () => {
                if !current_layers.is_empty() {
                    // Выделяем параметры для всего сегмента одним блоком.
                    let slices = {
                        let mut ps = param_store.lock().unwrap();
                        ps.allocate_segment(
                            &current_layer_sizes,
                            MemoryDeviceKind::HostRam, // начальное размещение
                        )
                    };
                    debug_assert_eq!(slices.len(), current_layers.len(),
                        "Number of slices must match number of layers");
                    segments.push(Segment::UniversalProcessor(
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

                    // Выделяем параметры для сегмента Splitter.
                    let slice = {
                        let mut ps = param_store.lock().unwrap();
                        let slices = ps.allocate_segment(
                            &[desc.param_len()],
                            MemoryDeviceKind::HostRam,
                        );
                        slices[0]
                    };
                    segments.push(Segment::Splitter {
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

                    // Выделяем параметры для сегмента Combiner.
                    let slice = {
                        let mut ps = param_store.lock().unwrap();
                        let slices = ps.allocate_segment(
                            &[desc.param_len()],
                            MemoryDeviceKind::HostRam,
                        );
                        slices[0]
                    };
                    segments.push(Segment::Combiner {
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
                    segments.push(Segment::Unsqueeze(target_dims));
                }
                LayerKind::ReduceMean => {
                    finalize_universal!();
                    let target_dims = desc.output_shape.streams.clone();
                    segments.push(Segment::ReduceMean(target_dims));
                }
                _ => {
                    if current_stream_indices.is_none() {
                        let indices = if let Some(ref ports) = active_ports {
                            if let Some(ref mut branch) = current_branch {
                                if let Some(pos) = ports.iter().position(|&p| p == desc.input_shape.streams[0]) {
                                    *branch = pos;
                                }
                            } else {
                                if let Some(pos) = ports.iter().position(|&p| p == desc.input_shape.streams[0]) {
                                    current_branch = Some(pos);
                                } else {
                                    current_branch = Some(0);
                                }
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

        let input_stream_count = match segments.first() {
            Some(Segment::CombinerConnector { input_dims, .. }) => input_dims.len(),
            _ => 1,
        };
        let output_stream_count = match segments.last() {
            Some(Segment::SplitterConnector { .. }) | Some(Segment::Splitter { .. }) => 2,
            _ => 1,
        };

        // -----------------------------------------------------------
        // 7. Вычисляем ожидаемые формы входных и выходных тензоров
        // -----------------------------------------------------------
        let input_shapes: Vec<Vec<usize>> = vec![layers.first().unwrap().input_shape.streams.clone()];
        let output_shapes: Vec<Vec<usize>> = if output_stream_count == 1 {
            vec![layers.last().unwrap().output_shape.streams.clone()]
        } else {
            let last_segment = segments.last().unwrap();
            match last_segment {
                Segment::Splitter { output_dims, .. } => {
                    output_dims.iter().map(|&d| vec![d]).collect()
                }
                Segment::SplitterConnector { dim_a, dim_b } => {
                    vec![vec![*dim_a], vec![*dim_b]]
                }
                _ => vec![],
            }
        };

        // -----------------------------------------------------------
        // 8. Создаём пул временных матриц
        // -----------------------------------------------------------
        let temp_matrix_pool = Arc::new(Mutex::new(TempMatrixPool::new(memory_executor.clone())));

        // -----------------------------------------------------------
        // 9. Выполняем начальное размещение сегментов
        // -----------------------------------------------------------
        compute_executor.redistribute(&segments, batch_size, true);

        // -----------------------------------------------------------
        // 10. Собираем MixedModel
        // -----------------------------------------------------------
        let model = MixedModel {
            segments,
            param_store,
            executor: cpu_executor,
            compute_executor,
            input_stream_count,
            output_stream_count,
            memory_executor,
            input_shapes,
            output_shapes,
            temp_matrix_pool,
            optimizer_exprs: HashMap::new(),
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