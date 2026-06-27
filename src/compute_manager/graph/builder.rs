// src/compute_manager/graph/builder.rs

use std::sync::{Arc, Mutex};

use crate::compute_manager::cpu::{CostModel, Scheduler, WorkerPool};
use crate::compute_manager::cpu::hardware::CPU_INFO;
use crate::compute_manager::cpu::scheduler::LayerInfo;
use crate::compute_manager::device::Device;
use crate::compute_manager::executor::Executor;
use crate::compute_manager::gpu::pipeline::PipelineCache;
use crate::compute_manager::gpu::GpuCompute;
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

// ---------- GPU-исполнитель (всегда компилируется) ----------
fn create_gpu_executor(device_index: usize) -> Result<Box<dyn Executor>, String> {
    let context = crate::compute_manager::gpu::init::create_gpu_context(device_index)?;
    Ok(Box::new(crate::compute_manager::gpu::executor::GpuExecutor::new(context)))
}

fn create_executor(device: Device) -> Result<(Box<dyn Executor>, Option<Mutex<GpuCompute>>), String> {
    match device {
        Device::Cpu { threads } => {
            let pool = Arc::new(WorkerPool::new(threads.max(1)));
            let cost = CostModel::calibrate();
            let mut sched = Scheduler::new(cost, CPU_INFO.clone());
            sched.set_num_workers(threads.max(1));
            Ok((Box::new(CpuExecutor::new(pool, Arc::new(Mutex::new(sched)))), None))
        }
        Device::Gpu { id } => {
            let executor = create_gpu_executor(id)?;
            let context = crate::compute_manager::gpu::init::create_gpu_context(id)
                .map_err(|e| format!("Failed to create GPU context: {}", e))?;
            let context = Arc::new(context);
            let device_arc = context.device.clone();
            let pipeline_cache = Arc::new(PipelineCache::new(device_arc));
            let gpu_compute = GpuCompute::new(context, pipeline_cache);
            Ok((executor, Some(Mutex::new(gpu_compute))))
        }
    }
}

impl MixedModel {
    pub fn from_plan(layers: Vec<LayerDesc>, num_threads: usize) -> Result<Self, String> {
        Self::from_plan_with_device(layers, num_threads, Device::Cpu { threads: num_threads })
    }

    pub fn from_plan_with_device(
        layers: Vec<LayerDesc>,
        _num_threads: usize,
        device: Device,
    ) -> Result<Self, String> {
        let store = Arc::new(Mutex::new(ParamStore::new()));
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
                    if !desc.out_features.is_empty() && !desc.in_features.is_empty() {
                        let dims = desc.out_features.clone();
                        assert_eq!(dims.len(), 2);
                        segments.push(Segment::SplitterConnector { dim_a: dims[0], dim_b: dims[1] });
                        active_ports = Some(dims);
                        current_branch = Some(0);
                    }
                }
                LayerKind::CombinerConnector => {
                    finalize_universal!();
                    if !desc.in_features.is_empty() && !desc.out_features.is_empty() {
                        segments.push(Segment::CombinerConnector {
                            input_dims: desc.in_features.clone(),
                            output_dim: desc.out_features[0],
                        });
                        active_ports = Some(vec![desc.out_features[0]]);
                        current_branch = None;
                    } else if !desc.out_features.is_empty() {
                        active_ports = Some(desc.out_features.clone());
                        current_branch = None;
                    }
                }
                LayerKind::Splitter => {
                    finalize_universal!();
                    let mut store_lock = store.lock().unwrap();
                    let slice = store_lock.allocate(desc.param_len());
                    drop(store_lock);
                    let p = desc.out_features[0];
                    let q = desc.out_features[1];
                    segments.push(Segment::Splitter {
                        input_dim: desc.in_features[0],
                        output_dims: vec![p, q],
                        slice,
                    });
                    active_ports = Some(vec![p, q]);
                    current_branch = Some(0);
                }
                LayerKind::Combiner => {
                    finalize_universal!();
                    let mut store_lock = store.lock().unwrap();
                    let slice = store_lock.allocate(desc.param_len());
                    drop(store_lock);
                    segments.push(Segment::Combiner {
                        input_dim: desc.in_features[0],
                        output_dim: desc.out_features[0],
                        slice,
                    });
                    active_ports = Some(vec![desc.out_features[0]]);
                    current_branch = None;
                }
                LayerKind::Unsqueeze(target_dims) => {
                    finalize_universal!();
                    segments.push(Segment::Unsqueeze(target_dims.clone()));
                }
                LayerKind::ReduceMean(target_dims) => {
                    finalize_universal!();
                    segments.push(Segment::ReduceMean(target_dims.clone()));
                }
                _ => {
                    if current_stream_indices.is_none() {
                        let indices = if let Some(ref ports) = active_ports {
                            if let Some(ref mut branch) = current_branch {
                                if let Some(pos) = ports.iter().position(|&p| p == desc.in_features[0]) {
                                    *branch = pos;
                                }
                            } else {
                                if let Some(pos) = ports.iter().position(|&p| p == desc.in_features[0]) {
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

        let num_cpu_threads = if let Device::Cpu { threads } = device { threads.max(1) } else { 2 };
        let cost = CostModel::calibrate();
        let mut scheduler = Scheduler::new(cost, CPU_INFO.clone());
        scheduler.set_num_workers(num_cpu_threads);
        let pool = Arc::new(WorkerPool::new(num_cpu_threads));

        let (executor, gpu_compute) = create_executor(device)?;

        Ok(MixedModel {
            segments,
            store,
            pool,
            scheduler: Mutex::new(scheduler),
            executor,
            gpu_compute,
            layer_infos,
            input_stream_count,
            output_stream_count,
        })
    }
}