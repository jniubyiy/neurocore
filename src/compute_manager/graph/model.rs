// src/compute_manager/graph/model.rs

use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};

use crate::compute_manager::compute_executor::ComputeExecutor;
use crate::compute_manager::dim_change::DynamicTensor;
use crate::compute_manager::executor::Executor;
use crate::compute_manager::graph::types::{ChunkedContexts, DynamicContext, Model};
use crate::compute_manager::matrix_buffer::{MatrixBufferHandle, TempMatrixPool};
use crate::compute_manager::memory_executor::MemoryExecutor;
use crate::compute_manager::memory_executor::types::MemoryDeviceKind;
use crate::device_plan::ComputeDevice;
use crate::loss_plan::{LossDesc, LossExpr};
use crate::model_plan::param_store::{ParamSlice, ParamStore};
use crate::optimizer_plan::{OptimizerDesc, OptimizerExpr};

pub struct MixedModel {
    pub(crate) models: Vec<Model>,
    pub(crate) param_store: Arc<Mutex<ParamStore>>,
    pub(crate) executor: Box<dyn Executor>,
    pub(crate) control_executor: Box<dyn Executor>,
    pub(crate) compute_executor: Arc<ComputeExecutor>,
    pub(crate) input_stream_count: usize,
    pub(crate) output_stream_count: usize,
    pub(crate) memory_executor: Arc<RwLock<MemoryExecutor>>,

    pub(crate) input_shapes: Vec<Vec<usize>>,
    pub(crate) output_shapes: Vec<Vec<usize>>,
    pub(crate) temp_matrix_pool: Arc<Mutex<TempMatrixPool>>,
    pub(crate) optimizer_exprs: HashMap<usize, OptimizerExpr>,
    pub(crate) last_forward_contexts: HashMap<usize, ChunkedContexts>,
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
        &self.param_store
    }

    pub fn executor(&self) -> &Box<dyn Executor> {
        &self.executor
    }

    pub fn control_executor(&self) -> &Box<dyn Executor> {
        &self.control_executor
    }

    pub fn memory_executor(&self) -> &Arc<RwLock<MemoryExecutor>> {
        &self.memory_executor
    }

    pub fn compute_executor(&self) -> &Arc<ComputeExecutor> {
        &self.compute_executor
    }

    pub fn models(&self) -> &[Model] {
        &self.models
    }

    pub fn forward(
        &mut self,
        input: DynamicTensor,
    ) -> (DynamicTensor, Vec<Vec<DynamicContext>>) {
        let pool_arc = self.temp_matrix_pool.clone();
        self.last_forward_contexts.clear();

        let buf = {
            let mut pool_guard = pool_arc.lock().unwrap();
            self.dynamic_tensor_to_buffer(&mut pool_guard, input)
        };

        let (out_bufs, _ctxs) = self.forward_mat_multi_buffered(pool_arc, vec![buf]);

        let out_buf = out_bufs.into_iter().next().expect("No output buffer");
        let out_tensor = self.buffer_to_dynamic_tensor(out_buf, &self.output_shapes[0]);

        (out_tensor, Vec::new())
    }

    pub fn backward(
        &mut self,
        delta: DynamicTensor,
    ) -> (DynamicTensor, Vec<Vec<f32>>) {
        let pool_arc = self.temp_matrix_pool.clone();

        let delta_buf = {
            let mut pool_guard = pool_arc.lock().unwrap();
            self.dynamic_tensor_to_buffer(&mut pool_guard, delta)
        };

        let in_bufs = self.backward_mat_multi_buffered(pool_arc, vec![delta_buf]);

        let ps = self.param_store.lock().unwrap();
        let total_params = ps.total_params();
        let mut grads_vec = vec![0.0f32; total_params];
        let mut offset = 0usize;
        for buffer_idx in 0..ps.num_buffers() {
            let buffer = ps.get_param_buffer_by_idx(buffer_idx);
            let grads_handle = &buffer.grads;
            let grads_data = self.read_buffer_to_vec(grads_handle);
            let len = grads_data.len();
            grads_vec[offset..offset + len].copy_from_slice(&grads_data);
            offset += len;
        }
        drop(ps);

        let in_buf = in_bufs.into_iter().next().expect("No input buffer");
        let in_tensor = self.buffer_to_dynamic_tensor(in_buf, &self.input_shapes[0]);

        (in_tensor, vec![grads_vec])
    }

    pub fn forward_multi(
        &mut self,
        inputs: Vec<DynamicTensor>,
    ) -> (Vec<DynamicTensor>, Vec<Vec<DynamicContext>>) {
        let pool_arc = self.temp_matrix_pool.clone();
        self.last_forward_contexts.clear();

        let bufs = {
            let mut pool_guard = pool_arc.lock().unwrap();
            let mut bufs = Vec::with_capacity(inputs.len());
            for tensor in inputs {
                bufs.push(self.dynamic_tensor_to_buffer(&mut pool_guard, tensor));
            }
            bufs
        };

        let (out_bufs, _ctxs) = self.forward_mat_multi_buffered(pool_arc, bufs);

        let out_tensors = out_bufs
            .into_iter()
            .zip(self.output_shapes.iter())
            .map(|(buf, shape)| self.buffer_to_dynamic_tensor(buf, shape))
            .collect();

        (out_tensors, Vec::new())
    }

    pub fn backward_multi(
        &mut self,
        deltas: Vec<DynamicTensor>,
    ) -> (Vec<DynamicTensor>, Vec<Vec<f32>>) {
        let pool_arc = self.temp_matrix_pool.clone();

        let delta_bufs = {
            let mut pool_guard = pool_arc.lock().unwrap();
            let mut bufs = Vec::with_capacity(deltas.len());
            for tensor in deltas {
                bufs.push(self.dynamic_tensor_to_buffer(&mut pool_guard, tensor));
            }
            bufs
        };

        let in_bufs = self.backward_mat_multi_buffered(pool_arc, delta_bufs);

        let ps = self.param_store.lock().unwrap();
        let total_params = ps.total_params();
        let mut grads_vec = vec![0.0f32; total_params];
        let mut offset = 0usize;
        for buffer_idx in 0..ps.num_buffers() {
            let buffer = ps.get_param_buffer_by_idx(buffer_idx);
            let grads_handle = &buffer.grads;
            let grads_data = self.read_buffer_to_vec(grads_handle);
            let len = grads_data.len();
            grads_vec[offset..offset + len].copy_from_slice(&grads_data);
            offset += len;
        }
        drop(ps);

        let in_tensors = in_bufs
            .into_iter()
            .zip(self.input_shapes.iter())
            .map(|(buf, shape)| self.buffer_to_dynamic_tensor(buf, shape))
            .collect();

        (in_tensors, vec![grads_vec])
    }

    pub fn compute_loss(
        &self,
        desc: LossDesc,
        pred: &DynamicTensor,
        target: &DynamicTensor,
    ) -> (f32, DynamicTensor) {
        let pool_arc = self.temp_matrix_pool.clone();
        let mut pool = pool_arc.lock().unwrap();

        let pred_buf = self.dynamic_tensor_to_buffer(&mut pool, pred.clone());
        let target_buf = self.dynamic_tensor_to_buffer(&mut pool, target.clone());

        let (loss, grad_buf) = self.compute_loss_handle(desc.build(), pred_buf, target_buf, &mut pool);

        let grad_tensor = self.buffer_to_dynamic_tensor(grad_buf, &self.output_shapes[0]);
        (loss, grad_tensor)
    }

    fn compute_loss_handle(
        &self,
        expr: Arc<LossExpr>,
        pred: MatrixBufferHandle,
        target: MatrixBufferHandle,
        pool: &mut TempMatrixPool,
    ) -> (f32, MatrixBufferHandle) {
        if self.compute_executor.has_gpu() {
            let gpu = self.compute_executor.gpu_compute().unwrap();
            if pred.is_gpu() && target.is_gpu() {
                return crate::plans::loss_plan::gpu_exec::compute_loss_gpu_buffered_handle(
                    &gpu, &expr, &pred, &target,
                );
            }
        }

        crate::plans::loss_plan::execution::compute_loss_mat_buffered(
            &expr, &pred, &target, pool,
        )
    }

    pub fn migrate_parameters(
        &mut self,
        placement: &[crate::compute_manager::compute_executor::ModelPlacement],
    ) -> Result<(), String> {
        let mut param_store = self.param_store.lock().unwrap();
        let memory_executor = self.memory_executor.clone();

        for (model_idx, model_placement) in placement.iter().enumerate() {
            let target_kind = match &model_placement.compute_device {
                ComputeDevice::Gpu { id } => {
                    MemoryDeviceKind::DeviceVram(crate::compute_manager::device_spec::DeviceId(*id))
                }
                ComputeDevice::Cpu { .. } => MemoryDeviceKind::HostRam,
            };

            let slices: Vec<ParamSlice> = match &self.models[model_idx] {
                Model::UniversalProcessor(_, param_slices, _) => param_slices.clone(),
                Model::Splitter { slice, .. } => vec![slice.clone()],
                Model::Combiner { slice, .. } => vec![slice.clone()],
                _ => Vec::new(),
            };

            if let Some(first_slice) = slices.first() {
                let buffer = param_store.get_param_buffer_mut(first_slice);
                if buffer.location != target_kind {
                    let mut mem = memory_executor.write().unwrap();
                    mem.move_matrix_handle(buffer.params.id(), target_kind).map_err(|e| {
                        format!("Failed to move params for model {}: {:?}", model_idx, e)
                    })?;
                    mem.move_matrix_handle(buffer.grads.id(), target_kind).map_err(|e| {
                        format!("Failed to move grads for model {}: {:?}", model_idx, e)
                    })?;
                    if let Some(ref opt_state) = buffer.opt_state {
                        mem.move_matrix_handle(opt_state.id(), target_kind).map_err(|e| {
                            format!("Failed to move opt_state for model {}: {:?}", model_idx, e)
                        })?;
                    }
                    buffer.location = target_kind;
                }
            }
        }
        Ok(())
    }

    fn dynamic_tensor_to_buffer(
        &self,
        pool: &mut TempMatrixPool,
        tensor: DynamicTensor,
    ) -> MatrixBufferHandle {
        let batch = tensor.batch_size();
        let features = tensor.features();
        let flat = tensor.to_flat();
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
        buf
    }

    fn buffer_to_dynamic_tensor(
        &self,
        buf: MatrixBufferHandle,
        shape: &[usize],
    ) -> DynamicTensor {
        if buf.is_gpu() {
            let gpu_compute = self
                .compute_executor
                .gpu_compute()
                .expect("GPU compute not available");
            let cpu_handle = gpu_compute.download_gpu_handle_to_cpu_handle(&buf);
            let vec = {
                let guard = cpu_handle.read();
                guard.as_slice().unwrap().to_vec()
            };
            let batch = buf.rows();
            let features = buf.cols();
            let mut flat = vec![0.0f32; batch * features];
            for r in 0..batch {
                for c in 0..features {
                    flat[r * features + c] = vec[c * batch + r];
                }
            }
            drop(buf);
            return self.flat_to_dynamic_tensor(shape, flat);
        }

        let batch = buf.rows();
        let features = buf.cols();
        let guard = buf.read();
        let slice = guard.as_slice().expect("CPU buffer");
        let mut flat = vec![0.0f32; batch * features];
        for r in 0..batch {
            for c in 0..features {
                flat[r * features + c] = slice[c * batch + r];
            }
        }
        drop(guard);
        drop(buf);
        self.flat_to_dynamic_tensor(shape, flat)
    }

    fn flat_to_dynamic_tensor(&self, shape: &[usize], flat: Vec<f32>) -> DynamicTensor {
        let feature_count: usize = shape.iter().product();
        assert_eq!(
            flat.len() % feature_count,
            0,
            "Flat size must be multiple of feature count"
        );
        let batch = flat.len() / feature_count;

        let mut dest = match shape.len() {
            1 => DynamicTensor::Dim1(crate::tensor::Tensor2D::zeros(batch, shape[0])),
            2 => DynamicTensor::Dim2(crate::tensor::Tensor3D::zeros(batch, shape[0], shape[1])),
            3 => DynamicTensor::Dim3(crate::tensor::Tensor4D::zeros(
                batch,
                shape[0],
                shape[1],
                shape[2],
            )),
            4 => DynamicTensor::Dim4(crate::tensor::Tensor5D::zeros(
                batch,
                shape[0],
                shape[1],
                shape[2],
                shape[3],
            )),
            _ => panic!(
                "Unsupported tensor dimensionality: {} spatial dims",
                shape.len()
            ),
        };
        let shape_ref = dest.clone();
        DynamicTensor::from_flat_into(&shape_ref, &flat, &mut dest);
        dest
    }

    fn read_buffer_to_vec(&self, handle: &MatrixBufferHandle) -> Vec<f32> {
        if handle.is_gpu() {
            let gpu_compute = self
                .compute_executor
                .gpu_compute()
                .expect("GPU compute not available");
            let cpu_handle = gpu_compute.download_gpu_handle_to_cpu_handle(handle);
            let guard = cpu_handle.read();
            guard.as_slice().unwrap().to_vec()
        } else {
            let guard = handle.read();
            guard.as_slice().expect("CPU buffer").to_vec()
        }
    }

    pub fn update_params_buffered(&mut self, desc: OptimizerDesc, _grads: &[f32]) {
        let mut ps = self.param_store.lock().unwrap();
        let gpu_compute_guard = self.compute_executor.gpu_compute();
        let gpu_compute_ref = gpu_compute_guard.as_deref();

        for buffer_idx in 0..ps.num_buffers() {
            let (num_params, slice) = {
                let buffer = ps.get_param_buffer_by_idx(buffer_idx);
                (
                    buffer.params.rows(),
                    ParamSlice::new(buffer_idx, 0, buffer.params.rows()),
                )
            };

            let chain = desc.build_chain();
            let state_size = chain.total_state_size_per_param();
            ps.ensure_opt_state(&slice, state_size);

            if !self.optimizer_exprs.contains_key(&buffer_idx) {
                let pool_arc = self.temp_matrix_pool.clone();
                let mut pool = pool_arc.lock().unwrap();
                let opt_expr = OptimizerExpr::new_buffered_handle(
                    self.memory_executor.clone(),
                    num_params,
                    chain,
                    &mut pool,
                );
                self.optimizer_exprs.insert(buffer_idx, opt_expr);
            }

            let opt = self.optimizer_exprs.get_mut(&buffer_idx).unwrap();
            let buffer = ps.get_param_buffer_by_idx(buffer_idx);
            let params_handle = buffer.params.clone();
            let grads_handle = buffer.grads.clone();

            opt.step_buffered_handle_hybrid(&params_handle, &grads_handle, gpu_compute_ref);
        }
    }

    pub(crate) fn get_params_handle_for_model(
        &self,
        model_idx: usize,
    ) -> Option<MatrixBufferHandle> {
        let ps = self.param_store.lock().unwrap();
        match &self.models[model_idx] {
            Model::UniversalProcessor(_, slices, _) => slices
                .first()
                .map(|s| ps.params_handle(s).clone()),
            Model::Splitter { slice, .. } | Model::Combiner { slice, .. } => {
                Some(ps.params_handle(slice).clone())
            }
            _ => None,
        }
    }

    pub(crate) fn get_grads_handle_for_model(
        &self,
        model_idx: usize,
    ) -> Option<MatrixBufferHandle> {
        let ps = self.param_store.lock().unwrap();
        match &self.models[model_idx] {
            Model::UniversalProcessor(_, slices, _) => slices
                .first()
                .map(|s| ps.grads_handle(s).clone()),
            Model::Splitter { slice, .. } | Model::Combiner { slice, .. } => {
                Some(ps.grads_handle(slice).clone())
            }
            _ => None,
        }
    }
}