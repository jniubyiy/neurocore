// src/compute_manager/graph/model.rs

use std::sync::{Arc, Mutex};

use crate::compute_manager::device_assignment::SegmentPlacement;
use crate::compute_manager::dim_change::DynamicTensor;
use crate::compute_manager::executor::Executor;
use crate::compute_manager::graph::types::{DynamicContext, Segment};
use crate::compute_manager::gpu::GpuCompute;
use crate::compute_manager::memory_executor::MemoryExecutor;
use crate::compute_manager::adaptive_planner::ProfilingData;
use crate::compute_manager::matrix_buffer::{MatrixBufferHandle, TempMatrixPool};
use crate::device_plan::DevicePlan;
use crate::loss_plan::{LossDesc, LossExpr};
use crate::model_plan::param_store::BufferedParamStore;
use crate::optimizer_plan::{OptimizerExpr, OptimizerDesc};

pub(crate) struct DevicePlacementState {
    pub(crate) profiling_data: ProfilingData,
    pub(crate) placements: Vec<SegmentPlacement>,
}

pub struct MixedModel {
    pub(crate) segments: Vec<Segment>,
    pub(crate) segment_placement: Vec<SegmentPlacement>,
    pub(crate) buffered_param_store: Arc<Mutex<BufferedParamStore>>,
    pub(crate) executor: Box<dyn Executor>,
    pub(crate) gpu_compute: Option<Mutex<GpuCompute>>,
    pub(crate) input_stream_count: usize,
    pub(crate) output_stream_count: usize,
    pub(crate) memory_executor: Arc<Mutex<MemoryExecutor>>,

    pub(crate) input_shapes: Vec<Vec<usize>>,
    pub(crate) output_shapes: Vec<Vec<usize>>,

    pub(crate) placement_state: Arc<Mutex<DevicePlacementState>>,

    /// Пул временных матриц для управляемого выделения памяти на CPU.
    pub(crate) temp_matrix_pool: Arc<Mutex<TempMatrixPool>>,

    /// Буферизованный интерпретатор оптимизатора с состояниями.
    pub(crate) optimizer_expr: Option<OptimizerExpr>,
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

    /// Возвращает доступ к буферизованному хранилищу параметров.
    pub fn buffered_param_store(&self) -> &Arc<Mutex<BufferedParamStore>> {
        &self.buffered_param_store
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
                self.allocate_and_set_placements(initial);
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
            self.allocate_and_set_placements(new_placements);
        }
    }

    fn allocate_and_set_placements(&mut self, new_placements: Vec<SegmentPlacement>) {
        let mut state = self.placement_state.lock().unwrap();
        state.placements = new_placements.clone();
        state.profiling_data = ProfilingData::new();
        self.segment_placement = new_placements;
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

    // ===================================================================
    // Публичные методы с DynamicTensor (конвертация на границе)
    // ===================================================================

    pub fn forward(
        &mut self,
        input: DynamicTensor,
    ) -> (DynamicTensor, Vec<Vec<DynamicContext>>) {
        let pool_arc = self.temp_matrix_pool.clone();
        let mut pool = pool_arc.lock().unwrap();

        let buf = self.dynamic_tensor_to_buffer(&mut pool, input);

        let (out_bufs, ctxs) = self.forward_mat_multi_buffered(&mut pool, vec![buf]);

        let out_buf = out_bufs.into_iter().next().expect("No output buffer");
        let out_tensor = self.buffer_to_dynamic_tensor(out_buf, &self.output_shapes[0]);

        (out_tensor, ctxs)
    }

    pub fn backward(
        &mut self,
        contexts: &[Vec<DynamicContext>],
        delta: DynamicTensor,
    ) -> (DynamicTensor, Vec<Vec<f32>>) {
        let pool_arc = self.temp_matrix_pool.clone();
        let mut pool = pool_arc.lock().unwrap();

        let delta_buf = self.dynamic_tensor_to_buffer(&mut pool, delta);

        let in_bufs = self.backward_mat_multi_buffered(&mut pool, contexts, vec![delta_buf]);

        // Получаем градиенты параметров из BufferedParamStore
        let bp = self.buffered_param_store.lock().unwrap();
        let mut grads_vec = vec![0.0f32; bp.len()];
        bp.copy_grads_to_slice(&mut grads_vec);
        drop(bp);

        let in_buf = in_bufs.into_iter().next().expect("No input buffer");
        let in_tensor = self.buffer_to_dynamic_tensor(in_buf, &self.input_shapes[0]);

        (in_tensor, vec![grads_vec])
    }

    pub fn forward_multi(
        &mut self,
        inputs: Vec<DynamicTensor>,
    ) -> (Vec<DynamicTensor>, Vec<Vec<DynamicContext>>) {
        let pool_arc = self.temp_matrix_pool.clone();
        let mut pool = pool_arc.lock().unwrap();

        let bufs: Vec<MatrixBufferHandle> = inputs
            .into_iter()
            .map(|tensor| self.dynamic_tensor_to_buffer(&mut pool, tensor))
            .collect();

        let (out_bufs, ctxs) = self.forward_mat_multi_buffered(&mut pool, bufs);

        let out_tensors = out_bufs
            .into_iter()
            .zip(self.output_shapes.iter())
            .map(|(buf, shape)| self.buffer_to_dynamic_tensor(buf, shape))
            .collect();

        (out_tensors, ctxs)
    }

    pub fn backward_multi(
        &mut self,
        contexts: &[Vec<DynamicContext>],
        deltas: Vec<DynamicTensor>,
    ) -> (Vec<DynamicTensor>, Vec<Vec<f32>>) {
        let pool_arc = self.temp_matrix_pool.clone();
        let mut pool = pool_arc.lock().unwrap();

        let delta_bufs: Vec<MatrixBufferHandle> = deltas
            .into_iter()
            .map(|tensor| self.dynamic_tensor_to_buffer(&mut pool, tensor))
            .collect();

        let in_bufs = self.backward_mat_multi_buffered(&mut pool, contexts, delta_bufs);

        let bp = self.buffered_param_store.lock().unwrap();
        let mut grads_vec = vec![0.0f32; bp.len()];
        bp.copy_grads_to_slice(&mut grads_vec);
        drop(bp);

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

    // Приватный метод: вычисление потерь полностью на MatrixBufferHandle
    fn compute_loss_handle(
        &self,
        expr: Arc<LossExpr>,
        pred: MatrixBufferHandle,
        target: MatrixBufferHandle,
        pool: &mut TempMatrixPool,
    ) -> (f32, MatrixBufferHandle) {
        if let Some(ref gpu_compute_mutex) = self.gpu_compute {
            let gpu = gpu_compute_mutex.lock().unwrap();
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

    // ===================================================================
    // Новые методы конвертации DynamicTensor <-> MatrixBufferHandle
    // ===================================================================

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
            let gpu_compute = self.gpu_compute.as_ref()
                .expect("GPU compute not available").lock().unwrap();
            let vec = gpu_compute.download_gpu_handle_to_vec(&buf);
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
        assert_eq!(flat.len() % feature_count, 0, "Flat size must be multiple of feature count");
        let batch = flat.len() / feature_count;

        let mut dest = match shape.len() {
            1 => DynamicTensor::Dim1(crate::tensor::Tensor2D::zeros(batch, shape[0])),
            2 => DynamicTensor::Dim2(crate::tensor::Tensor3D::zeros(batch, shape[0], shape[1])),
            3 => DynamicTensor::Dim3(crate::tensor::Tensor4D::zeros(batch, shape[0], shape[1], shape[2])),
            4 => DynamicTensor::Dim4(crate::tensor::Tensor5D::zeros(batch, shape[0], shape[1], shape[2], shape[3])),
            _ => panic!("Unsupported tensor dimensionality: {} spatial dims", shape.len()),
        };
        let shape_ref = dest.clone();
        DynamicTensor::from_flat_into(&shape_ref, &flat, &mut dest);
        dest
    }

    // ===================================================================
    // НОВЫЙ МЕТОД: шаг оптимизации через буферизованное хранилище
    // ===================================================================

    /// Выполняет один шаг оптимизации, используя `MatrixBufferHandle` и
    /// `MemoryExecutor`. Параметры и градиенты передаются через дескрипторы,
    /// состояние оптимизатора хранится в управляемой памяти.
    ///
    /// # Аргументы
    /// * `desc` – описание оптимизатора (цепочка кубиков).
    /// * `grads` – плоский срез градиентов для текущего шага.
    ///
    /// # Паника
    /// Паникует, если `grads.len()` не совпадает с числом параметров,
    /// или если оптимизатор не может быть инициализирован.
    pub fn update_params_buffered(&mut self, desc: OptimizerDesc, grads: &[f32]) {
        let chain = desc.build_chain();
        let state_size = chain.total_state_size_per_param();

        let mut bp = self.buffered_param_store.lock().unwrap();

        // Гарантируем, что состояние оптимизатора выделено правильно.
        bp.ensure_opt_state(state_size);

        // Копируем градиенты в управляемый буфер.
        bp.copy_grads_from_slice(grads);

        // Инициализация OptimizerExpr при первом вызове.
        if self.optimizer_expr.is_none() {
            let num_params = bp.len();
            let pool_arc = self.temp_matrix_pool.clone();
            let mut pool = pool_arc.lock().unwrap();
            let opt_expr = OptimizerExpr::new_buffered_handle(
                self.memory_executor.clone(),
                num_params,
                chain,
                &mut pool,
            );
            self.optimizer_expr = Some(opt_expr);
        }

        // Выполняем шаг оптимизатора над внутренними буферами.
        let opt = self.optimizer_expr.as_mut().unwrap();
        opt.step_buffered_handle(&bp.params, &bp.grads);

        // Никакого копирования обратно не требуется: параметры уже обновлены.
    }
}