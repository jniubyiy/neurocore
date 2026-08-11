// src/compute_manager/graph/model.rs

use std::sync::{Arc, Mutex};
use faer::Mat;

use crate::compute_manager::cpu::{Scheduler, WorkerPool};
use crate::compute_manager::cpu::scheduler::LayerInfo;
use crate::compute_manager::device_assignment::SegmentPlacement;
use crate::compute_manager::dim_change::DynamicTensor;
use crate::compute_manager::executor::Executor;
use crate::compute_manager::graph::types::{DynamicContext, Segment};
use crate::compute_manager::gpu::GpuCompute;
use crate::compute_manager::gpu::param_store::GpuParamStore;
use crate::compute_manager::memory_executor::MemoryExecutor;
use crate::compute_manager::persistent_buffer::SegmentPersistentBuffers;
use crate::compute_manager::adaptive_planner::ProfilingData;
use crate::device_plan::DevicePlan;
use crate::loss_plan::{LossDesc, LossExpr};
use crate::model_plan::param_store::ParamStore;
use crate::optimizer_plan::{OptimizerExpr, OptimizerChain, OptimizerDesc, cubes::*};
use crate::linalg;

pub(crate) struct DevicePlacementState {
    pub(crate) segment_buffers: Vec<Option<SegmentPersistentBuffers>>,
    pub(crate) profiling_data: ProfilingData,
    pub(crate) placements: Vec<SegmentPlacement>,
}

pub struct MixedModel {
    pub(crate) segments: Vec<Segment>,
    pub(crate) segment_placement: Vec<SegmentPlacement>,
    pub(crate) store: Arc<Mutex<ParamStore>>,
    pub(crate) pool: Arc<WorkerPool>,
    pub(crate) scheduler: Mutex<Scheduler>,
    pub(crate) executor: Box<dyn Executor>,
    pub(crate) gpu_compute: Option<Mutex<GpuCompute>>,
    pub(crate) gpu_param_store: Option<Mutex<GpuParamStore>>,
    #[allow(dead_code)]
    pub(crate) layer_infos: Vec<Vec<LayerInfo>>,
    pub(crate) input_stream_count: usize,
    pub(crate) output_stream_count: usize,
    pub(crate) memory_executor: Arc<Mutex<MemoryExecutor>>,

    pub(crate) input_shapes: Vec<Vec<usize>>,
    pub(crate) output_shapes: Vec<Vec<usize>>,

    pub(crate) placement_state: Arc<Mutex<DevicePlacementState>>,
}

impl MixedModel {
    pub fn num_workers(&self) -> usize {
        self.executor.num_workers()
    }

    pub fn input_stream_count(&self) -> usize {
        self.input_stream_count
    }

    pub fn output_stream_count(&self) -> usize {
        self.output_stream_count
    }

    pub fn param_store(&self) -> &Arc<Mutex<ParamStore>> {
        &self.store
    }

    pub fn executor(&self) -> &Box<dyn Executor> {
        &self.executor
    }

    pub fn memory_executor(&self) -> &Arc<Mutex<MemoryExecutor>> {
        &self.memory_executor
    }

    pub fn maybe_reassign_devices(&mut self, device_plan: &DevicePlan, batch_size: usize) {
        if self.segment_placement.is_empty() {
            return;
        }

        let (need_reassign, profiling_snapshot) = {
            let state = self.placement_state.lock().unwrap();
            if state.placements.is_empty() {
                let initial = self.segment_placement.clone();
                drop(state);
                self.allocate_and_set_placements(initial, batch_size);
                return;
            }
            let mut profiling = state.profiling_data.clone();
            let should = profiling.tick_and_should_reassign();
            (should, if should { Some(profiling) } else { None })
        };

        if need_reassign {
            let snapshot = profiling_snapshot.unwrap();
            let new_placements = {
                let (placements, _keep) = crate::compute_manager::adaptive_planner::assign_devices_adaptive(
                    &self.segments,
                    device_plan,
                    &mut self.memory_executor.lock().unwrap(),
                    batch_size,
                    Some(&snapshot),
                );
                placements
            };
            self.allocate_and_set_placements(new_placements, batch_size);
        }
    }

    fn allocate_and_set_placements(&mut self, new_placements: Vec<SegmentPlacement>, batch_size: usize) {
        if new_placements.is_empty() {
            return;
        }

        let mut state = self.placement_state.lock().unwrap();
        let n = self.segments.len();
        let mut old_buffers = std::mem::replace(&mut state.segment_buffers, vec![None; n]);
        // ensure old_buffers has length n (first call may give empty)
        if old_buffers.len() != n {
            old_buffers.resize(n, None);
        }

        let old_placements = std::mem::replace(&mut state.placements, new_placements.clone());
        let mut new_buffers = vec![None; n];

        let mut executor = self.memory_executor.lock().unwrap();

        for idx in 0..n {
            let old_pl = old_placements.get(idx);
            let new_pl = &new_placements[idx];
            if Some(new_pl) == old_pl {
                new_buffers[idx] = old_buffers[idx].clone();
            } else {
                if let Some(old_buf) = old_buffers[idx].clone() {
                    old_buf.release(&mut executor);
                }
                let buf = SegmentPersistentBuffers::for_segment(
                    &self.segments[idx],
                    &new_pl.compute_device,
                    batch_size,
                    &mut executor,
                );
                new_buffers[idx] = Some(buf);
            }
        }

        state.segment_buffers = new_buffers;
        state.placements = new_placements;
        state.profiling_data = ProfilingData::new();
        self.segment_placement = state.placements.clone();
    }

    pub(crate) fn record_segment_timing(
        &self,
        seg_index: usize,
        device: &crate::device_plan::plan::ComputeDevice,
        duration_ns: f64,
    ) {
        if let Ok(mut state) = self.placement_state.lock() {
            state.profiling_data.add(seg_index, device.clone(), duration_ns);
        }
    }

    pub(crate) fn get_segment_buffers(&self, seg_index: usize) -> Option<SegmentPersistentBuffers> {
        self.placement_state.lock().ok()?.segment_buffers[seg_index].clone()
    }

    pub fn create_optimizer(&self, chain: OptimizerChain) -> OptimizerExpr {
        let num_params = self.store.lock().unwrap().len();
        OptimizerExpr::new(num_params, chain)
    }

    pub fn update_params(&mut self, desc: OptimizerDesc, grads: &[f32]) {
        let chain = desc.build_chain();
        let mut opt = self.create_optimizer(chain);
        let mut store = self.store.lock().unwrap();
        let mut params = store.all_params_vec();
        opt.step(&mut params, grads);
        store.set_all_params(&params);
    }

    pub fn update_params_gpu(&self, desc: OptimizerDesc, step: usize) {
        let gpu_store = self.gpu_param_store
            .as_ref()
            .expect("GPU param store is not available");
        let gpu_compute_mutex = self.gpu_compute
            .as_ref()
            .expect("GPU compute is not available");
        let gpu_compute = gpu_compute_mutex.lock().unwrap();
        let mut store = gpu_store.lock().unwrap();

        let total = store.num_params;
        let chain = desc.build_chain();
        let cubes = chain.cubes();

        let required_state_per_param = chain.total_state_size_per_param();
        if required_state_per_param > 0 {
            let total_state_elems = total * required_state_per_param;
            let need_new = match &store.opt_state {
                None => true,
                Some(buf) => buf.len() < (total_state_elems as u64),
            };
            if need_new {
                let (new_state_buf, _state_id) = gpu_compute.create_buffer(
                    total_state_elems,
                    vulkano::buffer::BufferUsage::STORAGE_BUFFER | vulkano::buffer::BufferUsage::TRANSFER_DST,
                );
                store.opt_state = Some(new_state_buf);
            }
        }

        let mut state_offset = 0;

        for cube in cubes.iter() {
            let size_per_param = cube.state_size_per_param();

            let state_slice = store.opt_state.as_ref().map(|full_state| {
                let elem_size = std::mem::size_of::<f32>() as u64;
                let start_byte = (state_offset * total) as u64 * elem_size;
                let len_elems = size_per_param * total;
                let end_byte = start_byte + len_elems as u64 * elem_size;
                full_state.clone().slice(start_byte..end_byte)
            });

            if let Some(cube) = cube.as_any().downcast_ref::<ScaleGradient>() {
                gpu_compute.run_scale_gradient(&store.grads, cube.factor, total);
            } else if let Some(cube) = cube.as_any().downcast_ref::<AddWeightDecay>() {
                gpu_compute.run_weight_decay(&store.params, &store.grads, cube.decay, total);
            } else if let Some(cube) = cube.as_any().downcast_ref::<GradientClip>() {
                let min_val = cube.min.unwrap_or(f32::NEG_INFINITY);
                let max_val = cube.max.unwrap_or(f32::INFINITY);
                gpu_compute.run_gradient_clip(&store.grads, min_val, max_val, total);
            } else if let Some(cube) = cube.as_any().downcast_ref::<Momentum>() {
                if let Some(ref state) = state_slice {
                    gpu_compute.run_momentum(&store.grads, state, cube.beta, total);
                }
            } else if let Some(cube) = cube.as_any().downcast_ref::<NesterovMomentum>() {
                if let Some(ref state) = state_slice {
                    gpu_compute.run_nesterov_momentum(&store.grads, state, cube.beta, total);
                }
            } else if let Some(cube) = cube.as_any().downcast_ref::<AdamTransform>() {
                if let Some(ref state) = state_slice {
                    gpu_compute.run_adam(
                        &store.grads,
                        state,
                        cube.beta1,
                        cube.beta2,
                        cube.eps,
                        step,
                        total,
                    );
                }
            } else if cube.as_any().is::<ApplyUpdate>() {
                gpu_compute.run_apply_update(&store.params, &store.grads, total);
            } else {
                panic!("Unsupported optimizer cube for GPU: {:?}", std::any::type_name_of_val(cube));
            }

            state_offset += size_per_param;
        }
    }

    pub fn forward(
        &mut self,
        input: DynamicTensor,
    ) -> (DynamicTensor, Vec<Vec<DynamicContext>>) {
        let mat = self.dynamic_tensor_to_mat(input);
        let (out_mats, ctxs) = self.forward_mat_multi(&[mat]);
        let out_tensor = self.mat_to_dynamic_tensor(out_mats.into_iter().next().unwrap(), &self.output_shapes[0]);
        (out_tensor, ctxs)
    }

    pub fn backward(
        &mut self,
        contexts: &[Vec<DynamicContext>],
        delta: DynamicTensor,
    ) -> (DynamicTensor, Vec<Vec<f32>>) {
        let delta_mat = self.dynamic_tensor_to_mat(delta);
        let (in_mats, grads) = self.backward_mat_multi(contexts, &[delta_mat]);
        let in_tensor = self.mat_to_dynamic_tensor(in_mats.into_iter().next().unwrap(), &self.input_shapes[0]);
        (in_tensor, grads)
    }

    pub fn forward_multi(
        &mut self,
        inputs: Vec<DynamicTensor>,
    ) -> (Vec<DynamicTensor>, Vec<Vec<DynamicContext>>) {
        assert_eq!(inputs.len(), self.input_stream_count,
            "forward_multi: expected {} inputs, got {}", self.input_stream_count, inputs.len());

        let mats: Vec<Mat<f32>> = inputs.into_iter()
            .map(|t| self.dynamic_tensor_to_mat(t))
            .collect();

        let (out_mats, ctxs) = self.forward_mat_multi(&mats);

        let out_tensors = out_mats.into_iter()
            .zip(self.output_shapes.iter())
            .map(|(mat, shape)| self.mat_to_dynamic_tensor(mat, shape))
            .collect();

        if let Ok(mut exec) = self.memory_executor.lock() {
            exec.cleanup_temp_pools(std::time::Duration::from_secs(30));
        }

        (out_tensors, ctxs)
    }

    pub fn backward_multi(
        &mut self,
        contexts: &[Vec<DynamicContext>],
        deltas: Vec<DynamicTensor>,
    ) -> (Vec<DynamicTensor>, Vec<Vec<f32>>) {
        assert_eq!(deltas.len(), self.output_stream_count,
            "backward_multi: expected {} deltas, got {}", self.output_stream_count, deltas.len());

        let delta_mats: Vec<Mat<f32>> = deltas.into_iter()
            .map(|d| self.dynamic_tensor_to_mat(d))
            .collect();

        let (in_mats, grads) = self.backward_mat_multi(contexts, &delta_mats);

        let in_tensors = in_mats.into_iter()
            .zip(self.input_shapes.iter())
            .map(|(mat, shape)| self.mat_to_dynamic_tensor(mat, shape))
            .collect();

        (in_tensors, grads)
    }

    pub fn compute_loss(
        &self,
        desc: LossDesc,
        pred: &DynamicTensor,
        target: &DynamicTensor,
    ) -> (f32, DynamicTensor) {
        let pred_mat = self.dynamic_tensor_to_mat(pred.clone());
        let target_mat = self.dynamic_tensor_to_mat(target.clone());
        let (loss, grad_mat) = self.compute_loss_mat(desc.build(), &pred_mat, &target_mat);
        let grad_tensor = self.mat_to_dynamic_tensor(grad_mat, &self.output_shapes[0]);
        (loss, grad_tensor)
    }

    pub fn compute_loss_mat(
        &self,
        expr: Arc<LossExpr>,
        pred: &Mat<f32>,
        target: &Mat<f32>,
    ) -> (f32, Mat<f32>) {
        let use_gpu = self.gpu_compute.is_some() && (expr.num_tasks() == pred.nrows());
        if let Some(ref gpu_compute_mutex) = self.gpu_compute {
            if use_gpu {
                let gpu_compute = gpu_compute_mutex.lock().unwrap();
                crate::loss_plan::compute_loss_gpu(&gpu_compute, &expr, pred, target)
            } else {
                let mut scheduler = self.scheduler.lock().unwrap();
                crate::loss_plan::compute_loss_mat(&expr, pred, target, &mut scheduler, &self.pool)
            }
        } else {
            let mut scheduler = self.scheduler.lock().unwrap();
            crate::loss_plan::compute_loss_mat(&expr, pred, target, &mut scheduler, &self.pool)
        }
    }

    fn dynamic_tensor_to_mat(&self, tensor: DynamicTensor) -> Mat<f32> {
        match tensor {
            DynamicTensor::Dim1(t) => linalg::tensor2d_to_faer(&t),
            DynamicTensor::Dim2(t) => linalg::tensor3d_to_faer(&t),
            DynamicTensor::Dim3(t) => linalg::tensor4d_to_faer(&t),
            DynamicTensor::Dim4(t) => linalg::tensor5d_to_faer(&t),
        }
    }

    fn mat_to_dynamic_tensor(&self, mat: Mat<f32>, shape: &[usize]) -> DynamicTensor {
        let batch = mat.nrows();
        match shape.len() {
            1 => DynamicTensor::Dim1(linalg::faer_to_tensor2d(&mat)),
            2 => DynamicTensor::Dim2(linalg::faer_to_tensor3d(&mat, batch, shape[0], shape[1])),
            3 => DynamicTensor::Dim3(linalg::faer_to_tensor4d(&mat, batch, shape[0], shape[1], shape[2])),
            4 => DynamicTensor::Dim4(linalg::faer_to_tensor5d(&mat, batch, shape[0], shape[1], shape[2], shape[3])),
            _ => panic!("Unsupported tensor dimensionality: {} spatial dims", shape.len()),
        }
    }

    pub fn forward_universal_batch_mat(
        layers: &[Box<dyn crate::layers::UniversalLayer>],
        slices: &[crate::model_plan::param_store::ParamSlice],
        batch: &Mat<f32>,
        params: &[f32],
    ) -> (Mat<f32>, Vec<DynamicContext>) {
        let mut current = batch.clone();
        let mut ctxs = Vec::new();
        for (layer, slice) in layers.iter().zip(slices.iter()) {
            let (next, ctx) = layer.forward_mat(&current, params, slice);
            ctxs.push(ctx);
            current = next;
        }
        (current, ctxs)
    }
}