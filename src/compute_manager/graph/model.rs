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
use crate::compute_manager::memory_executor::MemoryExecutor;
use crate::compute_manager::adaptive_planner::ProfilingData;
use crate::compute_manager::matrix_buffer::{MatrixBufferHandle, TempMatrixPool};
use crate::device_plan::DevicePlan;
use crate::loss_plan::{LossDesc, LossExpr};
use crate::model_plan::param_store::{BufferedParamStore, ParamStore};
use crate::optimizer_plan::{OptimizerExpr, OptimizerChain, OptimizerDesc, cubes::*};
use crate::linalg;

pub(crate) struct DevicePlacementState {
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
    #[allow(dead_code)]
    pub(crate) layer_infos: Vec<Vec<LayerInfo>>,
    pub(crate) input_stream_count: usize,
    pub(crate) output_stream_count: usize,
    pub(crate) memory_executor: Arc<Mutex<MemoryExecutor>>,

    pub(crate) input_shapes: Vec<Vec<usize>>,
    pub(crate) output_shapes: Vec<Vec<usize>>,

    pub(crate) placement_state: Arc<Mutex<DevicePlacementState>>,

    /// Пул временных матриц для управляемого выделения памяти на CPU.
    /// Обёрнут в Arc<Mutex<...>> для потокобезопасного доступа из графа.
    pub(crate) temp_matrix_pool: Arc<Mutex<TempMatrixPool>>,

    /// Новое буферизованное хранилище параметров и градиентов.
    /// Использует `MatrixBufferHandle`.
    pub(crate) buffered_param_store: Option<BufferedParamStore>,

    /// Буферизованный интерпретатор оптимизатора с состояниями.
    /// Создаётся лениво и хранится между шагами обучения.
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

    pub fn create_optimizer(&self, chain: OptimizerChain) -> OptimizerExpr {
        let num_params = self.store.lock().unwrap().len();
        OptimizerExpr::new(num_params, chain)
    }

    /// Устаревший метод обновления параметров на CPU через срезы `Vec<f32>`.
    /// Оставлен для обратной совместимости; рекомендуется использовать
    /// `update_params_buffered`.
    #[deprecated(note = "Use update_params_buffered for MemoryExecutor integration")]
    pub fn update_params(&mut self, desc: OptimizerDesc, grads: &[f32]) {
        let chain = desc.build_chain();
        let mut opt = self.create_optimizer(chain);
        let mut store = self.store.lock().unwrap();
        let mut params = store.all_params_vec();
        opt.step(&mut params, grads);
        store.set_all_params(&params);
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

        let mat = self.dynamic_tensor_to_mat(input);
        let buf = self.mat_to_matrix_buffer_handle(&mat, &mut pool);

        let (out_bufs, ctxs) = self.forward_mat_multi_buffered(&mut pool, vec![buf]);
        let out_buf = out_bufs.into_iter().next().expect("No output buffer");
        let out_tensor = self.matrix_buffer_handle_to_dynamic_tensor(out_buf, &self.output_shapes[0]);

        (out_tensor, ctxs)
    }

    pub fn backward(
        &mut self,
        contexts: &[Vec<DynamicContext>],
        delta: DynamicTensor,
    ) -> (DynamicTensor, Vec<Vec<f32>>) {
        let pool_arc = self.temp_matrix_pool.clone();
        let mut pool = pool_arc.lock().unwrap();

        let delta_mat = self.dynamic_tensor_to_mat(delta);
        let delta_buf = self.mat_to_matrix_buffer_handle(&delta_mat, &mut pool);

        let (in_bufs, grads) = self.backward_mat_multi_buffered(&mut pool, contexts, vec![delta_buf]);
        let in_buf = in_bufs.into_iter().next().expect("No input buffer");
        let in_tensor = self.matrix_buffer_handle_to_dynamic_tensor(in_buf, &self.input_shapes[0]);

        (in_tensor, grads)
    }

    pub fn forward_multi(
        &mut self,
        inputs: Vec<DynamicTensor>,
    ) -> (Vec<DynamicTensor>, Vec<Vec<DynamicContext>>) {
        let pool_arc = self.temp_matrix_pool.clone();
        let mut pool = pool_arc.lock().unwrap();

        let bufs: Vec<MatrixBufferHandle> = inputs
            .into_iter()
            .map(|tensor| {
                let mat = self.dynamic_tensor_to_mat(tensor);
                self.mat_to_matrix_buffer_handle(&mat, &mut pool)
            })
            .collect();

        let (out_bufs, ctxs) = self.forward_mat_multi_buffered(&mut pool, bufs);

        let out_tensors = out_bufs
            .into_iter()
            .zip(self.output_shapes.iter())
            .map(|(buf, shape)| self.matrix_buffer_handle_to_dynamic_tensor(buf, shape))
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
            .map(|tensor| {
                let mat = self.dynamic_tensor_to_mat(tensor);
                self.mat_to_matrix_buffer_handle(&mat, &mut pool)
            })
            .collect();

        let (in_bufs, grads) = self.backward_mat_multi_buffered(&mut pool, contexts, delta_bufs);

        let in_tensors = in_bufs
            .into_iter()
            .zip(self.input_shapes.iter())
            .map(|(buf, shape)| self.matrix_buffer_handle_to_dynamic_tensor(buf, shape))
            .collect();

        (in_tensors, grads)
    }

    pub fn compute_loss(
        &self,
        desc: LossDesc,
        pred: &DynamicTensor,
        target: &DynamicTensor,
    ) -> (f32, DynamicTensor) {
        let pool_arc = self.temp_matrix_pool.clone();
        let mut pool = pool_arc.lock().unwrap();

        let pred_mat = self.dynamic_tensor_to_mat(pred.clone());
        let target_mat = self.dynamic_tensor_to_mat(target.clone());
        let pred_buf = self.mat_to_matrix_buffer_handle(&pred_mat, &mut pool);
        let target_buf = self.mat_to_matrix_buffer_handle(&target_mat, &mut pool);

        let (loss, grad_buf) = self.compute_loss_handle(desc.build(), pred_buf, target_buf, &mut pool);

        let grad_mat = if grad_buf.is_gpu() {
            let gpu_compute = self.gpu_compute.as_ref().expect("GPU compute not available").lock().unwrap();
            gpu_compute.download_gpu_handle_to_mat(&grad_buf)
        } else {
            let guard = grad_buf.read();
            let slice = guard.as_slice().expect("CPU buffer");
            Mat::from_fn(grad_buf.rows(), grad_buf.cols(), |r, c| slice[c * grad_buf.rows() + r])
        };
        let grad_tensor = self.mat_to_dynamic_tensor(grad_mat, &self.output_shapes[0]);
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

    #[deprecated(note = "Use compute_loss (DynamicTensor) or compute_loss_handle internally")]
    pub fn compute_loss_mat(
        &self,
        expr: Arc<LossExpr>,
        pred: &Mat<f32>,
        target: &Mat<f32>,
    ) -> (f32, Mat<f32>) {
        let pool_arc = self.temp_matrix_pool.clone();
        let mut pool = pool_arc.lock().unwrap();

        let pred_buf = self.mat_to_matrix_buffer_handle(pred, &mut pool);
        let target_buf = self.mat_to_matrix_buffer_handle(target, &mut pool);

        let (loss, grad_buf) = self.compute_loss_handle(expr, pred_buf, target_buf, &mut pool);

        let grad_mat = if grad_buf.is_gpu() {
            let gpu = self.gpu_compute.as_ref().expect("GPU compute not available").lock().unwrap();
            gpu.download_gpu_handle_to_mat(&grad_buf)
        } else {
            let guard = grad_buf.read();
            let slice = guard.as_slice().expect("CPU buffer");
            Mat::from_fn(grad_buf.rows(), grad_buf.cols(), |r, c| slice[c * grad_buf.rows() + r])
        };
        (loss, grad_mat)
    }

    // ===================================================================
    // Вспомогательные функции конвертации DynamicTensor <-> MatrixBufferHandle
    // ===================================================================

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

    fn mat_to_matrix_buffer_handle(&self, mat: &Mat<f32>, pool: &mut TempMatrixPool) -> MatrixBufferHandle {
        let buf = pool.acquire(mat.nrows(), mat.ncols());
        {
            let mut guard = buf.write();
            let slice = guard.as_slice_mut().expect("CPU buffer");
            for c in 0..mat.ncols() {
                for r in 0..mat.nrows() {
                    slice[c * mat.nrows() + r] = mat[(r, c)];
                }
            }
        }
        buf
    }

    fn matrix_buffer_handle_to_dynamic_tensor(
        &self,
        buf: MatrixBufferHandle,
        shape: &[usize],
    ) -> DynamicTensor {
        let mat = if buf.is_gpu() {
            let gpu_compute = self.gpu_compute.as_ref().expect("GPU compute not available").lock().unwrap();
            gpu_compute.download_gpu_handle_to_mat(&buf)
        } else {
            let guard = buf.read();
            let slice = guard.as_slice().expect("CPU buffer");
            Mat::from_fn(buf.rows(), buf.cols(), |r, c| slice[c * buf.rows() + r])
        };

        let tensor = self.mat_to_dynamic_tensor(mat, shape);
        drop(buf);
        tensor
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

        // Инициализация BufferedParamStore при первом вызове
        if self.buffered_param_store.is_none() {
            let num_params = self.store.lock().unwrap().len();
            let bp = BufferedParamStore::new_cpu(
                self.memory_executor.clone(),
                num_params,
                state_size,
            );
            self.buffered_param_store = Some(bp);
        }

        // Инициализация OptimizerExpr при первом вызове
        if self.optimizer_expr.is_none() {
            let num_params = self.buffered_param_store.as_ref().unwrap().num_params();
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

        let bp = self.buffered_param_store.as_mut().unwrap();

        // Копируем параметры из ParamStore в управляемый буфер
        let params_vec = self.store.lock().unwrap().all_params_vec();
        bp.copy_params_from_slice(&params_vec);

        // Копируем градиенты
        bp.copy_grads_from_slice(grads);

        // Выполняем шаг оптимизатора
        let opt = self.optimizer_expr.as_mut().unwrap();
        opt.step_buffered_handle(bp.params_handle(), bp.grads_handle());

        // Копируем обновлённые параметры обратно в ParamStore
        let mut updated_params = vec![0.0; bp.num_params()];
        bp.copy_params_to_slice(&mut updated_params);
        self.store.lock().unwrap().set_all_params(&updated_params);
    }
}