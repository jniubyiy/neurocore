// src/compute_manager/graph/model.rs

use std::sync::{Arc, Mutex};

use faer::Mat;

use crate::compute_manager::cpu::{Scheduler, WorkerPool};
use crate::compute_manager::cpu::scheduler::LayerInfo;
use crate::compute_manager::dim_change::DynamicTensor;
use crate::compute_manager::executor::Executor;
use crate::compute_manager::graph::types::{DynamicContext, Segment};
use crate::compute_manager::gpu::GpuCompute;
use crate::loss_plan::{LossDesc, LossExpr};
use crate::model_plan::param_store::ParamStore;
use crate::optimizer_plan::{OptimizerExpr, OptimizerChain, OptimizerDesc};
use crate::linalg;

pub struct MixedModel {
    pub(crate) segments: Vec<Segment>,
    pub(crate) store: Arc<Mutex<ParamStore>>,
    pub(crate) pool: Arc<WorkerPool>,                    // используется для compute_loss_mat (CPU)
    pub(crate) scheduler: Mutex<Scheduler>,              // используется для compute_loss_mat (CPU)
    pub(crate) executor: Box<dyn Executor>,              // универсальный исполнитель (CPU/GPU)
    pub(crate) gpu_compute: Option<Mutex<GpuCompute>>,  // реальные GPU-вычисления (если устройство GPU)
    #[allow(dead_code)]
    pub(crate) layer_infos: Vec<Vec<LayerInfo>>,
    pub(crate) input_stream_count: usize,
    pub(crate) output_stream_count: usize,
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

    // ── Одиночные вход/выход (обратная совместимость) ──
    pub fn forward(
        &self,
        input: DynamicTensor,
    ) -> (DynamicTensor, Vec<Vec<DynamicContext>>) {
        let (outs, ctxs) = self.forward_multi(vec![input]);
        assert_eq!(outs.len(), 1);
        (outs.into_iter().next().unwrap(), ctxs)
    }

    pub fn backward(
        &self,
        contexts: &[Vec<DynamicContext>],
        delta: DynamicTensor,
    ) -> (DynamicTensor, Vec<Vec<f32>>) {
        let (ins, grads) = self.backward_multi(contexts, vec![delta]);
        assert_eq!(ins.len(), 1);
        (ins.into_iter().next().unwrap(), grads)
    }

    // ── Множественные входы/выходы (публичное API) ──
    pub fn forward_multi(
        &self,
        inputs: Vec<DynamicTensor>,
    ) -> (Vec<DynamicTensor>, Vec<Vec<DynamicContext>>) {
        assert_eq!(inputs.len(), self.input_stream_count,
            "forward_multi: expected {} inputs, got {}", self.input_stream_count, inputs.len());

        let mats: Vec<Mat<f32>> = inputs.iter().map(|t| match t {
            DynamicTensor::Dim1(t) => linalg::tensor2d_to_faer(t),
            DynamicTensor::Dim2(t) => linalg::tensor3d_to_faer(t),
            DynamicTensor::Dim3(t) => linalg::tensor4d_to_faer(t),
            DynamicTensor::Dim4(t) => linalg::tensor5d_to_faer(t),
        }).collect();

        let (out_mats, ctxs) = self.forward_mat_multi(&mats);

        let out_tensors = out_mats.into_iter().zip(inputs.iter()).map(|(mat, original)| {
            match original {
                DynamicTensor::Dim1(_) => DynamicTensor::Dim1(linalg::faer_to_tensor2d(&mat)),
                DynamicTensor::Dim2(t) => DynamicTensor::Dim2(
                    linalg::faer_to_tensor3d(&mat, t.dim1, t.dim2, mat.ncols()),
                ),
                DynamicTensor::Dim3(t) => DynamicTensor::Dim3(
                    linalg::faer_to_tensor4d(&mat, t.dim1, t.dim2, t.dim3, mat.ncols()),
                ),
                DynamicTensor::Dim4(t) => DynamicTensor::Dim4(
                    linalg::faer_to_tensor5d(&mat, t.dim1, t.dim2, t.dim3, t.dim4, mat.ncols()),
                ),
            }
        }).collect();
        (out_tensors, ctxs)
    }

    pub fn backward_multi(
        &self,
        contexts: &[Vec<DynamicContext>],
        deltas: Vec<DynamicTensor>,
    ) -> (Vec<DynamicTensor>, Vec<Vec<f32>>) {
        assert_eq!(deltas.len(), self.output_stream_count,
            "backward_multi: expected {} deltas, got {}", self.output_stream_count, deltas.len());

        let delta_mats: Vec<Mat<f32>> = deltas.iter().map(|d| match d {
            DynamicTensor::Dim1(t) => linalg::tensor2d_to_faer(t),
            DynamicTensor::Dim2(t) => linalg::tensor3d_to_faer(t),
            DynamicTensor::Dim3(t) => linalg::tensor4d_to_faer(t),
            DynamicTensor::Dim4(t) => linalg::tensor5d_to_faer(t),
        }).collect();

        let (in_mats, grads) = self.backward_mat_multi(contexts, &delta_mats);

        let in_tensors = in_mats.into_iter().zip(deltas.iter()).map(|(mat, original)| {
            match original {
                DynamicTensor::Dim1(_) => DynamicTensor::Dim1(linalg::faer_to_tensor2d(&mat)),
                DynamicTensor::Dim2(t) => DynamicTensor::Dim2(
                    linalg::faer_to_tensor3d(&mat, t.dim1, t.dim2, mat.ncols()),
                ),
                DynamicTensor::Dim3(t) => DynamicTensor::Dim3(
                    linalg::faer_to_tensor4d(&mat, t.dim1, t.dim2, t.dim3, mat.ncols()),
                ),
                DynamicTensor::Dim4(t) => DynamicTensor::Dim4(
                    linalg::faer_to_tensor5d(&mat, t.dim1, t.dim2, t.dim3, t.dim4, mat.ncols()),
                ),
            }
        }).collect();
        (in_tensors, grads)
    }

    // ── Потери ──
    pub fn compute_loss(
        &self,
        desc: LossDesc,
        pred: &DynamicTensor,
        target: &DynamicTensor,
    ) -> (f32, DynamicTensor) {
        let expr = desc.build();
        let pred_mat = match pred {
            DynamicTensor::Dim1(t) => linalg::tensor2d_to_faer(t),
            DynamicTensor::Dim2(t) => linalg::tensor3d_to_faer(t),
            DynamicTensor::Dim3(t) => linalg::tensor4d_to_faer(t),
            DynamicTensor::Dim4(t) => linalg::tensor5d_to_faer(t),
        };
        let target_mat = match target {
            DynamicTensor::Dim1(t) => linalg::tensor2d_to_faer(t),
            DynamicTensor::Dim2(t) => linalg::tensor3d_to_faer(t),
            DynamicTensor::Dim3(t) => linalg::tensor4d_to_faer(t),
            DynamicTensor::Dim4(t) => linalg::tensor5d_to_faer(t),
        };
        let (loss, grad_mat) = self.compute_loss_mat(expr, &pred_mat, &target_mat);
        let grad_tensor = DynamicTensor::Dim1(linalg::faer_to_tensor2d(&grad_mat));
        (loss, grad_tensor)
    }

    pub fn compute_loss_mat(
        &self,
        expr: Arc<LossExpr>,
        pred: &Mat<f32>,
        target: &Mat<f32>,
    ) -> (f32, Mat<f32>) {
        let mut scheduler = self.scheduler.lock().unwrap();
        crate::loss_plan::compute_loss_mat(&expr, pred, target, &mut scheduler, &self.pool)
    }

    // ── Вспомогательные ──
    pub fn samples_to_mat(samples: &[DynamicTensor]) -> Mat<f32> {
        if samples.is_empty() {
            panic!("samples_to_mat: empty slice");
        }
        let first = &samples[0];
        let features = match first {
            DynamicTensor::Dim1(t) => t.dim2,
            _ => panic!("samples_to_mat: only Dim1 supported"),
        };
        let batch = samples.len();
        let mut mat = Mat::zeros(batch, features);
        for (i, sample) in samples.iter().enumerate() {
            match sample {
                DynamicTensor::Dim1(t) => {
                    for (j, &val) in t.data[0].iter().enumerate() {
                        mat[(i, j)] = val;
                    }
                }
                _ => panic!("Inconsistent sample dimensions in samples_to_mat"),
            }
        }
        mat
    }

    pub fn mat_to_samples(mat: &Mat<f32>) -> Vec<DynamicTensor> {
        let batch = mat.nrows();
        let features = mat.ncols();
        let mut samples = Vec::with_capacity(batch);
        for i in 0..batch {
            let row: Vec<f32> = (0..features).map(|j| mat[(i, j)]).collect();
            samples.push(DynamicTensor::Dim1(crate::tensor::Tensor2D::new(vec![row])));
        }
        samples
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