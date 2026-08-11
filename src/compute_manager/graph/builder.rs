// src/compute_manager/graph/builder.rs
//
// Построитель вычислительного графа MixedModel.
// Все внутренние операции используют только матрицы faer::Mat<f32>.
// Тензоры (DynamicTensor) применяются исключительно на публичных границах
// (методы forward/backward обёртки), которые реализованы в model.rs.

use std::sync::{Arc, Mutex};

use crate::compute_manager::cpu::{CostModel, Scheduler, WorkerPool};
use crate::compute_manager::cpu::hardware::CPU_INFO;
use crate::compute_manager::cpu::scheduler::LayerInfo;
use crate::compute_manager::device::Device;
use crate::compute_manager::device_assignment::{assign_devices, SegmentPlacement};
use crate::compute_manager::device_spec::DeviceId;
use crate::compute_manager::executor::Executor;
use crate::compute_manager::gpu::pipeline::PipelineCache;
use crate::compute_manager::gpu::GpuCompute;
use crate::compute_manager::gpu::param_store::GpuParamStore;
use crate::compute_manager::memory_executor::MemoryExecutor;
use crate::device_plan::{ComputeDevice, DevicePlan};
use crate::layers::UniversalLayer;
use crate::model_plan::layer_desc::LayerDesc;
use crate::model_plan::blueprint::LayerKind;
use crate::model_plan::param_store::{ParamSlice, ParamStore};

use super::model::MixedModel;
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

    /// Основной конструктор с планом устройств и размером батча (для учёта памяти активаций).
    pub(crate) fn from_plan_with_device_plan_and_batch(
        layers: Vec<LayerDesc>,
        device_plan: DevicePlan,
        batch_size: usize,
    ) -> Result<Self, String> {
        // -----------------------------------------------------------
        // 1. Создаём MemoryExecutor и получаем GPU-контекст
        // -----------------------------------------------------------
        let (memory_executor, gpu_context) = device_plan.build_memory_executor();

        eprintln!("[BUILDER] gpu_context is_some = {}", gpu_context.is_some());

        let mut mem_exec = memory_executor.lock().unwrap();

        // -----------------------------------------------------------
        // 2. Суммарное количество потоков CPU
        // -----------------------------------------------------------
        let cpu_threads: usize = device_plan.compute_devices.iter()
            .filter_map(|d| match d {
                ComputeDevice::Cpu { threads, .. } => Some(*threads),
                _ => None,
            })
            .sum();
        let cpu_threads = cpu_threads.max(1);

        // Количество CPU (для mini‑model)
        let num_cpus = device_plan.compute_devices.iter()
            .filter(|d| matches!(d, ComputeDevice::Cpu { .. }))
            .count()
            .max(1);

        // -----------------------------------------------------------
        // 3. Параметры и планировщик CPU
        // -----------------------------------------------------------
        let store = Arc::new(Mutex::new(ParamStore::new()));
        let cost = CostModel::calibrate();
        let mut scheduler = Scheduler::new_with_cpus(cost, CPU_INFO.clone(), num_cpus);
        scheduler.set_num_workers(cpu_threads);
        let pool = Arc::new(WorkerPool::new(cpu_threads));
        let cpu_executor: Box<dyn Executor> = Box::new(CpuExecutor::new(pool.clone(), Arc::new(Mutex::new(scheduler.clone()))));

        // -----------------------------------------------------------
        // 4. Настройка исполнителя (CPU или GPU)
        // -----------------------------------------------------------
        let mut gpu_compute: Option<Mutex<GpuCompute>> = None;
        let mut gpu_param_store: Option<Mutex<GpuParamStore>> = None;
        let executor: Box<dyn Executor> = if let Some(gpu_ctx) = gpu_context {
            let gpu_id = device_plan.compute_devices.iter()
                .find_map(|d| if let ComputeDevice::Gpu { id } = d { Some(*id) } else { None })
                .unwrap_or(0);
            let gpu_executor = crate::compute_manager::gpu::GpuExecutor::new(gpu_ctx.as_ref().clone());
            let pipeline_cache = Arc::new(PipelineCache::new(gpu_ctx.device.clone()));

            eprintln!("[BUILDER] PipelineCache created successfully");

            let gpu_compute_instance = GpuCompute::new(
                gpu_ctx,
                pipeline_cache,
                memory_executor.clone(),
                DeviceId(gpu_id),
            );

            eprintln!("[BUILDER] GpuCompute created successfully");

            gpu_compute = Some(Mutex::new(gpu_compute_instance));
            Box::new(gpu_executor)
        } else {
            eprintln!("[BUILDER] No GPU context, falling back to CPU");
            cpu_executor.clone_executor()
        };

        // -----------------------------------------------------------
        // 5. Строим сегменты модели
        // -----------------------------------------------------------
        let mut segments: Vec<Segment> = Vec::new();
        let mut layer_infos: Vec<Vec<LayerInfo>> = Vec::new();
        let mut current_layers: Vec<Box<dyn UniversalLayer>> = Vec::new();
        let mut current_slices: Vec<ParamSlice> = Vec::new();
        let mut active_ports: Option<Vec<usize>> = None;
        let mut current_branch: Option<usize> = None;
        let mut current_stream_indices: Option<Vec<usize>> = None;

        macro_rules! finalize_universal {
            () => {
                if !current_layers.is_empty() {
                    let infos: Vec<LayerInfo> = current_layers
                        .iter()
                        .enumerate()
                        .map(|(i, layer)| LayerInfo {
                            id: i,
                            layer_type: crate::compute_manager::cpu::scheduler::LayerType::Linear,
                            in_features: layer.input_features(),
                            out_features: layer.output_features(),
                            total_rows: 0,
                        })
                        .collect();
                    segments.push(Segment::UniversalProcessor(
                        Arc::new(std::mem::take(&mut current_layers)),
                        std::mem::take(&mut current_slices),
                        current_stream_indices.take(),
                    ));
                    layer_infos.push(infos);
                }
            };
        }

        for desc in &layers {
            match &desc.kind {
                LayerKind::SplitterConnector => {
                    finalize_universal!();
                    let dims = &desc.output_shape.streams;
                    assert_eq!(dims.len(), 2);
                    segments.push(Segment::SplitterConnector {
                        dim_a: dims[0],
                        dim_b: dims[1],
                    });
                    active_ports = Some(dims.clone());
                    current_branch = Some(0);
                }
                LayerKind::CombinerConnector => {
                    finalize_universal!();
                    let input_dims = &desc.input_shape.streams;
                    let output_dim = desc.output_shape.streams[0];
                    segments.push(Segment::CombinerConnector {
                        input_dims: input_dims.clone(),
                        output_dim,
                    });
                    active_ports = Some(vec![output_dim]);
                    current_branch = None;
                }
                LayerKind::Splitter => {
                    finalize_universal!();
                    let input_dim = desc.input_shape.streams[0];
                    let output_dims = desc.output_shape.streams.clone();
                    active_ports = Some(output_dims.clone());
                    let mut store_lock = store.lock().unwrap();
                    let slice = store_lock.allocate(desc.param_len());
                    drop(store_lock);
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
                    let mut store_lock = store.lock().unwrap();
                    let slice = store_lock.allocate(desc.param_len());
                    drop(store_lock);
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
                    // Обычный слой: определяем, в каком потоке он выполняется
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
                    let mut store_lock = store.lock().unwrap();
                    let layer = desc.create_universal_layer();
                    let slice = store_lock.allocate(desc.param_len());
                    current_layers.push(layer);
                    current_slices.push(slice);
                    drop(store_lock);
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
        // 5.5 Назначаем устройства сегментам с учётом памяти
        // -----------------------------------------------------------
        let segment_placement = assign_devices(&segments, &device_plan, &mut mem_exec, batch_size)
            .map_err(|e| format!("Ошибка распределения устройств с учётом памяти: {}", e))?;

        drop(mem_exec);

        // -----------------------------------------------------------
        // 6. Создаём GPU-хранилище параметров, если есть GPU
        // -----------------------------------------------------------
        if let Some(ref gpu_compute_mutex) = gpu_compute {
            let initial_params = store.lock().unwrap().all_params_vec();
            let gpu_compute = gpu_compute_mutex.lock().unwrap();
            let gpu_store = GpuParamStore::from_cpu(
                gpu_compute.context.memory_allocator.clone(),
                &initial_params,
                0,
            );
            gpu_param_store = Some(Mutex::new(gpu_store));
            eprintln!("[BUILDER] GpuParamStore initialized");
        }

        // -----------------------------------------------------------
        // 7. Вычисляем ожидаемые формы входных и выходных тензоров
        //    (используются только для восстановления формы при тензорных обёртках)
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
                _ => {
                    vec![]
                }
            }
        };

        eprintln!(
            "[BUILDER] gpu_compute.is_some() = {}, gpu_param_store.is_some() = {}",
            gpu_compute.is_some(),
            gpu_param_store.is_some()
        );

        // -----------------------------------------------------------
        // 8. Собираем MixedModel
        // -----------------------------------------------------------
        Ok(MixedModel {
            segments,
            segment_placement,
            store,
            pool,
            scheduler: Mutex::new(scheduler),
            executor,
            gpu_compute,
            gpu_param_store,
            layer_infos,
            input_stream_count,
            output_stream_count,
            memory_executor,
            input_shapes,
            output_shapes,
        })
    }

    /// Обратно-совместимый конструктор (без batch_size, использует 1).
    pub(crate) fn from_plan_with_device_plan(
        layers: Vec<LayerDesc>,
        device_plan: DevicePlan,
    ) -> Result<Self, String> {
        Self::from_plan_with_device_plan_and_batch(layers, device_plan, 1)
    }

    /// Сборка модели с планом устройств (вызывается из публичного API).
    pub(crate) fn build_with_device_plan(
        plan: crate::model_plan::plan::Plan,
        device_plan: DevicePlan,
    ) -> Result<Self, String> {
        Self::from_plan_with_device_plan_and_batch(plan.layers, device_plan, 1)
    }

    /// Сборка модели с планом устройств и указанием размера батча.
    pub(crate) fn build_with_device_plan_and_batch(
        plan: crate::model_plan::plan::Plan,
        device_plan: DevicePlan,
        batch_size: usize,
    ) -> Result<Self, String> {
        Self::from_plan_with_device_plan_and_batch(plan.layers, device_plan, batch_size)
    }
}