// src/compute_manager/graph_v2/model_v2.rs

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use once_cell::sync::Lazy;

use crate::compute_manager::core::dim_change::DynamicTensor;
use crate::compute_manager::core::dynamic_context::DynamicContext;
use crate::compute_manager::distributor_v2::{
    SegmentTopologyInfo, SmartDistributor,
};
use crate::compute_manager::jobs_v2::{
    ForwardContextsV2,
    Job, JobResult, LossJob,
    OptimizerApplyUpdateJob, OptimizerModifyGradsJob,
    ParamInitJob,
};
use crate::compute_manager::operators_v2::gpu_v2::compute::GpuCompute;
use crate::compute_manager::operators_v2::memory_v2::buffer::{MatrixBufferHandle, TempMatrixPool};
use crate::compute_manager::operators_v2::memory_v2::types::MemoryDeviceKind;
use crate::device_plan::DevicePlan;
use crate::layers::adapter::{
    AdapterCallStats, AdapterContext, AdapterPassStats,
};
use crate::loss_plan::desc::LossDesc;
use crate::model_plan::adapter_store::AdapterStateStore;
use crate::model_plan::layer_desc::LayerDesc;
use crate::model_plan::param_store::ParamStore;
use crate::optimizer_plan::OptimizerDesc;
use crate::training_plan::Initializer;

use super::builder_v2::build_segments;
use super::forward_v2::run_forward;
use super::backward_v2::run_backward;
use super::observer_v2::{
    EpochReportV2, GraphObserverV2, MonitorConfigV2, WarningV2,
};
use super::types_v2::{
    ForwardCacheV2, SegmentForwardStateV2, SegmentKindV2, SegmentV2,
};

static DEBUG_ADAPTER: Lazy<bool> =
    Lazy::new(|| std::env::var("NEUROCORE_DEBUG_ADAPTER").is_ok());

static DEBUG_ADAPTER_VERBOSE: Lazy<bool> =
    Lazy::new(|| std::env::var("NEUROCORE_DEBUG_ADAPTER_VERBOSE").is_ok());

static DISABLE_ADAPTERS: Lazy<bool> =
    Lazy::new(|| std::env::var("NEUROCORE_DISABLE_ADAPTERS").is_ok());

static ADAPTER_PASS_CALLS: AtomicUsize = AtomicUsize::new(0);

const ADAPTER_TRACE_FIRST_N: usize = 5;
const ADAPTER_TRACE_EVERY_N: usize = 50;

pub struct GraphV2 {
    pub(crate) segments: Vec<SegmentV2>,
    pub(crate) distributor: Arc<SmartDistributor>,
    pub(crate) observer: GraphObserverV2,
    pub(crate) param_store: Arc<Mutex<ParamStore>>,
    pub(crate) adapter_store: Arc<Mutex<AdapterStateStore>>,
    pub(crate) temp_pool: Arc<Mutex<TempMatrixPool>>,
    pub(crate) input_shape: Vec<usize>,
    pub(crate) output_shape: Vec<usize>,
    pub(crate) forward_cache: Option<ForwardCacheV2>,
    pub(crate) seed: Option<u64>,
}

impl GraphV2 {
    pub fn build(
        layers_desc: Vec<LayerDesc>,
        device_plan: &DevicePlan,
    ) -> Result<Self, String> {
        if layers_desc.is_empty() {
            return Err("GraphV2::build: empty layer list".into());
        }

        let distributor = Arc::new(SmartDistributor::new(device_plan)?);

        let memory = distributor.snapshot().memory_executor.clone();

        let param_store = Arc::new(Mutex::new(ParamStore::new(memory.clone())));
        let adapter_store = Arc::new(Mutex::new(AdapterStateStore::new(memory)));

        let segments = build_segments(
            &layers_desc,
            &param_store,
            MemoryDeviceKind::HostRam,
        )?;

        let input_shape = layers_desc
            .first()
            .map(|l| l.input_shape.streams.clone())
            .unwrap_or_default();
        let output_shape = layers_desc
            .last()
            .map(|l| l.output_shape.streams.clone())
            .unwrap_or_default();

        let temp_pool = distributor.temp_pool();

        Ok(Self {
            segments,
            distributor,
            observer: GraphObserverV2::new(),
            param_store,
            adapter_store,
            temp_pool,
            input_shape,
            output_shape,
            forward_cache: None,
            seed: None,
        })
    }

    pub fn with_monitor_config(mut self, cfg: MonitorConfigV2) -> Self {
        self.observer = GraphObserverV2::with_config(cfg);
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    pub fn forward(&mut self, input: DynamicTensor) -> Result<DynamicTensor, String> {
        self.forward_cache = None;

        // Извлекаем sample_lens ДО конвертации в handle.
        let sample_lens: Option<Vec<usize>> = input.sample_lens().map(|s| s.to_vec());

        let input_handle = {
            let mut pool = self.temp_pool.lock().unwrap();
            dynamic_tensor_to_handle(&mut *pool, input)?
        };

        let (_out_handle, cache) = run_forward(
            &self.distributor,
            &self.param_store,
            &self.segments,
            input_handle,
            sample_lens.clone(),
        )?;

        let out_handle_cpu = self.copy_handle_to_host_ram(&cache.output)?;

        self.forward_cache = Some(cache);

        let out_tensor =
            handle_to_dynamic_tensor(&out_handle_cpu, &self.output_shape)?;
        Ok(out_tensor)
    }

    pub fn loss(
        &self,
        loss_desc: LossDesc,
        pred: &DynamicTensor,
        target: &DynamicTensor,
    ) -> Result<(f32, DynamicTensor), String> {
        let (pred_handle, target_handle) = {
            let mut pool = self.temp_pool.lock().unwrap();
            let p = dynamic_tensor_to_handle(&mut *pool, pred.clone())?;
            let t = dynamic_tensor_to_handle(&mut *pool, target.clone())?;
            (p, t)
        };

        let expr = loss_desc.build();
        let job = Job::Loss(LossJob {
            expr,
            pred: pred_handle,
            target: target_handle,
        });

        let result = self.distributor.dispatch(job);
        match result {
            JobResult::Loss { value, grad_pred } => {
                let grad_pred_cpu = self.copy_handle_to_host_ram(&grad_pred)?;

                let grad_tensor =
                    handle_to_dynamic_tensor(&grad_pred_cpu, &self.output_shape)?;
                Ok((value, grad_tensor))
            }
            JobResult::Failed(msg) => Err(format!("GraphV2::loss: {}", msg)),
            _ => Err("GraphV2::loss: unexpected JobResult".into()),
        }
    }

    pub fn backward(&mut self, delta: DynamicTensor) -> Result<DynamicTensor, String> {
        let cache = self
            .forward_cache
            .take()
            .ok_or_else(|| "GraphV2::backward: forward_cache is empty".to_string())?;

        let delta_handle = {
            let mut pool = self.temp_pool.lock().unwrap();
            dynamic_tensor_to_handle(&mut *pool, delta)?
        };

        let grad_input = run_backward(
            &self.distributor,
            &self.param_store,
            &self.segments,
            &cache,
            delta_handle,
        )?;

        self.forward_cache = Some(cache);

        let grad_input_cpu = self.copy_handle_to_host_ram(&grad_input)?;

        let grad_tensor =
            handle_to_dynamic_tensor(&grad_input_cpu, &self.input_shape)?;
        Ok(grad_tensor)
    }

    pub fn optimizer_modify_grads(&self, optimizer: OptimizerDesc) -> Result<(), String> {
        let num_buffers = {
            let ps = self.param_store.lock().unwrap();
            ps.num_buffers()
        };

        for buffer_idx in 0..num_buffers {
            let (params, grads) = {
                let ps = self.param_store.lock().unwrap();
                let buffer = ps.get_param_buffer_by_idx(buffer_idx);
                (buffer.params.clone(), buffer.grads.clone())
            };

            let job = Job::OptimizerModifyGrads(OptimizerModifyGradsJob {
                buffer_idx,
                params,
                grads,
                optimizer: optimizer.clone(),
                gpu_compute: None,
            });

            match self.distributor.dispatch(job) {
                JobResult::Unit => {}
                JobResult::Failed(msg) => {
                    return Err(format!(
                        "GraphV2::optimizer_modify_grads: buffer {}: {}",
                        buffer_idx, msg
                    ));
                }
                _ => {
                    return Err(format!(
                        "GraphV2::optimizer_modify_grads: buffer {} unexpected JobResult",
                        buffer_idx
                    ));
                }
            }
        }
        Ok(())
    }

    pub fn adapter_pass(&self) -> Result<AdapterPassStats, String> {
        if *DISABLE_ADAPTERS {
            return Ok(AdapterPassStats::default());
        }

        let cache = match self.forward_cache.as_ref() {
            Some(c) => c,
            None => {
                return Err(
                    "GraphV2::adapter_pass: forward_cache is empty. \
                     adapter_pass must be called after forward/loss/backward \
                     and before optimizer_apply_update (MIGRATION_PLAN.md §2, I-1)."
                        .to_string(),
                );
            }
        };

        let batch = cache.batch;
        let debug = *DEBUG_ADAPTER;
        let verbose = debug && *DEBUG_ADAPTER_VERBOSE;
        let gpu_opt = self.distributor.gpu_compute();

        if cache.segment_states.len() != self.segments.len() {
            return Err(format!(
                "GraphV2::adapter_pass: cache size ({}) != segments count ({})",
                cache.segment_states.len(),
                self.segments.len()
            ));
        }

        let mut pass_stats = AdapterPassStats::default();

        for (seg, state) in self.segments.iter().zip(cache.segment_states.iter()) {
            let SegmentKindV2::Universal { layers, slices } = &seg.kind else {
                continue;
            };
            let SegmentForwardStateV2::Universal { contexts } = state else {
                continue;
            };

            let contexts_local: Vec<DynamicContext> = match contexts {
                ForwardContextsV2::Sequential(v) => v.clone(),
                ForwardContextsV2::Chunked { contexts, .. } => {
                    contexts.first().cloned().unwrap_or_default()
                }
            };

            let (params_handle, grads_handle) = {
                let ps = self.param_store.lock().unwrap();
                let first_slice = slices.first().ok_or_else(|| {
                    format!(
                        "GraphV2::adapter_pass: segment {} has no slices",
                        seg.index
                    )
                })?;
                (
                    ps.params_handle(first_slice).clone(),
                    ps.grads_handle(first_slice).clone(),
                )
            };

            for (i, layer) in layers.iter().enumerate() {
                let Some(adapter) = layer.adapter() else {
                    continue;
                };

                let own_slice = slices[i];
                let forward_ctx_ref: Option<&DynamicContext> = contexts_local.get(i);

                let before_l2 = if debug {
                    compute_grads_l2(&grads_handle, own_slice, gpu_opt.as_deref())
                } else {
                    f32::NAN
                };

                let ctx = AdapterContext {
                    segment_params: &params_handle,
                    segment_grads: &grads_handle,
                    own_slice,
                    all_slices: slices,
                    batch,
                    optimizer_applied: true,
                    own_state_slice: None,
                    adapter_store: Some(self.adapter_store.clone()),
                    forward_ctx: forward_ctx_ref,
                };

                adapter.apply(&ctx);

                let after_l2 = if debug {
                    compute_grads_l2(&grads_handle, own_slice, gpu_opt.as_deref())
                } else {
                    f32::NAN
                };

                if debug {
                    let scale = if before_l2.abs() > 1e-30 && before_l2.is_finite()
                        && after_l2.is_finite()
                    {
                        after_l2 / before_l2
                    } else {
                        0.0
                    };

                    let call_idx = ADAPTER_PASS_CALLS.fetch_add(1, Ordering::Relaxed);
                    let should_print = verbose
                        || call_idx < ADAPTER_TRACE_FIRST_N
                        || call_idx % ADAPTER_TRACE_EVERY_N == 0;

                    if should_print {
                        if verbose {
                            eprintln!(
                                "[ADAPTER {}] #{} seg={} layer={} \
                                 before_l2={:.6e} after_l2={:.6e} scale={:.6}",
                                adapter.name(),
                                call_idx,
                                seg.index,
                                i,
                                before_l2,
                                after_l2,
                                scale
                            );
                        } else {
                            eprintln!(
                                "[ADAPTER {}] #{} seg={} layer={} \
                                 before_l2={:.3e} after_l2={:.3e} scale={:.4}",
                                adapter.name(),
                                call_idx,
                                seg.index,
                                i,
                                before_l2,
                                after_l2,
                                scale
                            );
                        }
                    }
                }

                pass_stats.calls.push(AdapterCallStats {
                    name: adapter.name(),
                    before_l2,
                    after_l2,
                });
            }
        }

        Ok(pass_stats)
    }

    pub fn optimizer_apply_update(&mut self, optimizer: OptimizerDesc) -> Result<(), String> {
        let num_buffers = {
            let ps = self.param_store.lock().unwrap();
            ps.num_buffers()
        };

        for buffer_idx in 0..num_buffers {
            let (params, grads) = {
                let ps = self.param_store.lock().unwrap();
                let buffer = ps.get_param_buffer_by_idx(buffer_idx);
                (buffer.params.clone(), buffer.grads.clone())
            };

            let job = Job::OptimizerApplyUpdate(OptimizerApplyUpdateJob {
                buffer_idx,
                params,
                grads,
                optimizer: optimizer.clone(),
                gpu_compute: None,
            });

            match self.distributor.dispatch(job) {
                JobResult::Unit => {}
                JobResult::Failed(msg) => {
                    return Err(format!(
                        "GraphV2::optimizer_apply_update: buffer {}: {}",
                        buffer_idx, msg
                    ));
                }
                _ => {
                    return Err(format!(
                        "GraphV2::optimizer_apply_update: buffer {} unexpected JobResult",
                        buffer_idx
                    ));
                }
            }
        }

        self.forward_cache = None;
        Ok(())
    }

    pub fn optimizer_step(&mut self, optimizer: OptimizerDesc) -> Result<(), String> {
        self.optimizer_modify_grads(optimizer.clone())?;
        let _stats = self.adapter_pass()?;
        self.optimizer_apply_update(optimizer)?;
        Ok(())
    }

    pub fn observe_step(&mut self, loss: f32, grad_norm: Option<f32>) {
        self.observer.record_step(loss, grad_norm);
    }

    pub fn observe_epoch(&mut self, epoch: usize) -> Result<(), String> {
        let should = epoch == 0 || self.observer.should_reassign();
        if should {
            let infos = self.collect_segment_infos();
            self.distributor.on_epoch_boundary(&infos)?;
            self.observer.mark_reassigned();
        }
        Ok(())
    }

    pub fn end_epoch(&mut self) -> EpochReportV2 {
        self.observer.end_epoch()
    }

    pub fn all_warnings(&self) -> &[WarningV2] {
        self.observer.all_warnings()
    }

    pub fn init_params(&self, initializer: Initializer) -> Result<(), String> {
        let all_data: Vec<f32> = {
            let ps = self.param_store.lock().unwrap();
            let total = ps.total_params();

            if total == 0 {
                Vec::new()
            } else {
                match &initializer {
                    Initializer::Zeros => vec![0.0; total],
                    Initializer::Ones => vec![1.0; total],
                    Initializer::RandomUniform { min, max } => {
                        use rand::rngs::StdRng;
                        use rand::{Rng, SeedableRng};
                        let mut rng: Box<dyn rand::RngCore> = match self.seed {
                            Some(s) => Box::new(StdRng::seed_from_u64(s)),
                            None => Box::new(rand::thread_rng()),
                        };
                        (0..total).map(|_| rng.gen_range(*min..*max)).collect()
                    }
                }
            }
        };

        let num_buffers = {
            let ps = self.param_store.lock().unwrap();
            ps.num_buffers()
        };

        let mut offset = 0usize;
        for buffer_idx in 0..num_buffers {
            let (params, len) = {
                let ps = self.param_store.lock().unwrap();
                let b = ps.get_param_buffer_by_idx(buffer_idx);
                (b.params.clone(), b.params.rows() * b.params.cols())
            };

            if len == 0 {
                continue;
            }

            let slice = all_data[offset..offset + len].to_vec();
            offset += len;

            let job = Job::ParamInit(ParamInitJob {
                buffer_idx,
                params,
                initializer: initializer.clone(),
                seed: self.seed,
                precomputed: Some(slice),
            });

            match self.distributor.dispatch(job) {
                JobResult::Unit => {}
                JobResult::Failed(msg) => {
                    return Err(format!(
                        "GraphV2::init_params: buffer {}: {}",
                        buffer_idx, msg
                    ));
                }
                _ => {
                    return Err(format!(
                        "GraphV2::init_params: buffer {} unexpected JobResult",
                        buffer_idx
                    ));
                }
            }
        }
        Ok(())
    }

    pub fn distributor(&self) -> &Arc<SmartDistributor> {
        &self.distributor
    }

    pub fn param_store(&self) -> &Arc<Mutex<ParamStore>> {
        &self.param_store
    }

    pub fn adapter_store(&self) -> &Arc<Mutex<AdapterStateStore>> {
        &self.adapter_store
    }

    pub fn num_segments(&self) -> usize {
        self.segments.len()
    }

    pub fn input_shape(&self) -> &[usize] {
        &self.input_shape
    }

    pub fn output_shape(&self) -> &[usize] {
        &self.output_shape
    }

    fn copy_handle_to_host_ram(
        &self,
        handle: &MatrixBufferHandle,
    ) -> Result<MatrixBufferHandle, String> {
        let cpu_buf = {
            let mut pool = self.temp_pool.lock().unwrap();
            pool.acquire(handle.rows(), handle.cols())
        };

        if handle.is_gpu() {
            let gpu = self
                .distributor
                .gpu_compute()
                .ok_or_else(|| {
                    "GraphV2::copy_handle_to_host_ram: GPU handle but no GpuCompute"
                        .to_string()
                })?;
            gpu.copy_gpu_to_cpu_handle(handle, &cpu_buf);
        } else {
            let src = handle.read();
            let src_slice = src
                .as_slice()
                .expect("GraphV2::copy_handle_to_host_ram: expected CPU buffer");
            cpu_buf.write_range(0, src_slice);
        }

        Ok(cpu_buf)
    }

    fn collect_segment_infos(&self) -> Vec<SegmentTopologyInfo> {
        let ps = self.param_store.lock().unwrap();
        let _as = self.adapter_store.lock().unwrap();

        let mut infos = Vec::with_capacity(self.segments.len());
        for seg in &self.segments {
            match &seg.kind {
                SegmentKindV2::Universal { layers, slices } => {
                    let param_count: usize = layers.iter().map(|l| l.param_len()).sum();
                    let (param_handle, grad_handle, opt_state_handle, current_loc) =
                        match slices.first() {
                            Some(s) => {
                                let buffer = ps.get_param_buffer(s);
                                (
                                    Some(buffer.params.clone()),
                                    Some(buffer.grads.clone()),
                                    buffer.opt_state.clone(),
                                    buffer.location,
                                )
                            }
                            None => (None, None, None, MemoryDeviceKind::HostRam),
                        };
                    infos.push(SegmentTopologyInfo {
                        index: seg.index,
                        param_count,
                        param_handle,
                        grad_handle,
                        opt_state_handle,
                        adapter_state_handle: None,
                        current_param_location: current_loc,
                    });
                }
                _ => {
                    infos.push(SegmentTopologyInfo {
                        index: seg.index,
                        param_count: 0,
                        param_handle: None,
                        grad_handle: None,
                        opt_state_handle: None,
                        adapter_state_handle: None,
                        current_param_location: MemoryDeviceKind::HostRam,
                    });
                }
            }
        }
        infos
    }
}

// ============================================================================
// Вспомогательные функции
// ============================================================================

fn compute_grads_l2(
    grads: &MatrixBufferHandle,
    slice: crate::model_plan::param_store::ParamSlice,
    gpu: Option<&GpuCompute>,
) -> f32 {
    if slice.len == 0 {
        return 0.0;
    }

    let values: Vec<f32> = if grads.is_gpu() {
        match gpu {
            Some(g) => {
                let full = g.download_gpu_handle_to_vec(grads);
                if slice.end() <= full.len() {
                    full[slice.start..slice.end()].to_vec()
                } else {
                    Vec::new()
                }
            }
            None => Vec::new(),
        }
    } else {
        grads.read_range(slice.start, slice.len)
    };

    let mut sum_sq = 0.0f64;
    for &v in &values {
        let d = v as f64;
        sum_sq += d * d;
    }
    (sum_sq.sqrt()) as f32
}

// ============================================================================
// Конвертации DynamicTensor ↔ MatrixBufferHandle
// ============================================================================

pub(crate) fn dynamic_tensor_to_handle(
    pool: &mut TempMatrixPool,
    tensor: DynamicTensor,
) -> Result<MatrixBufferHandle, String> {
    let batch = tensor.batch_size();
    let features = tensor.features();
    let flat = tensor.to_flat();
    if flat.len() != batch * features {
        return Err(format!(
            "dynamic_tensor_to_handle: flat len {} != batch*features {}",
            flat.len(),
            batch * features
        ));
    }

    let buf = pool.acquire(batch, features);
    {
        let mut guard = buf.write();
        let slice = guard.as_slice_mut().expect("CPU buffer");
        for r in 0..batch {
            for c in 0..features {
                slice[c * batch + r] = flat[r * features + c];
            }
        }
    }
    Ok(buf)
}

pub(crate) fn handle_to_dynamic_tensor(
    handle: &MatrixBufferHandle,
    shape: &[usize],
) -> Result<DynamicTensor, String> {
    if handle.is_gpu() {
        return Err(
            "handle_to_dynamic_tensor: GPU handles not supported (copy to HostRam first)"
                .into(),
        );
    }

    let batch = handle.rows();
    let features = handle.cols();

    let feature_count: usize = shape.iter().product();
    if feature_count != features {
        return Err(format!(
            "handle_to_dynamic_tensor: shape {:?} product {} != cols {}",
            shape, feature_count, features
        ));
    }

    let guard = handle.read();
    let slice = guard.as_slice().expect("CPU buffer");
    let mut flat = vec![0.0f32; batch * features];
    for r in 0..batch {
        for c in 0..features {
            flat[r * features + c] = slice[c * batch + r];
        }
    }
    drop(guard);

    flat_to_dynamic_tensor(shape, flat)
}

fn flat_to_dynamic_tensor(shape: &[usize], flat: Vec<f32>) -> Result<DynamicTensor, String> {
    let feature_count: usize = shape.iter().product();
    if feature_count == 0 {
        return Err("flat_to_dynamic_tensor: zero feature_count".into());
    }
    if flat.len() % feature_count != 0 {
        return Err(format!(
            "flat_to_dynamic_tensor: flat len {} not divisible by feature_count {}",
            flat.len(),
            feature_count
        ));
    }
    let batch = flat.len() / feature_count;

    let dest = match shape.len() {
        1 => DynamicTensor::Dim1(crate::tensor::Tensor2D::zeros(batch, shape[0])),
        2 => DynamicTensor::Dim2(crate::tensor::Tensor3D::zeros(batch, shape[0], shape[1])),
        3 => DynamicTensor::Dim3(crate::tensor::Tensor4D::zeros(
            batch, shape[0], shape[1], shape[2],
        )),
        4 => DynamicTensor::Dim4(crate::tensor::Tensor5D::zeros(
            batch, shape[0], shape[1], shape[2], shape[3],
        )),
        _ => {
            return Err(format!(
                "flat_to_dynamic_tensor: unsupported spatial dims {}",
                shape.len()
            ));
        }
    };

    Ok(DynamicTensor::from_flat(&dest, flat))
}