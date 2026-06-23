// src/layers/combiner/combiner.rs

use crate::compute_manager::dim_change::DynamicTensor;
use crate::compute_manager::graph::types::DynamicContext;
use crate::model_plan::param_store::ParamSlice;
use crate::linalg;
use faer::Mat;
use crate::neuron::Combiner as CombinerNeuron;

pub struct Combiner {
    input_dim: usize,
    output_dim: usize,
}

impl Combiner {
    pub fn new(input_dims: Vec<usize>, output_dim: usize) -> Self {
        assert_eq!(input_dims.len(), 2, "Combiner требует ровно два входа");
        assert!(input_dims[0] == input_dims[1], "Combiner пока требует одинаковых размеров входов");
        Self { input_dim: input_dims[0], output_dim }
    }

    pub fn input_dims(&self) -> Vec<usize> { vec![self.input_dim, self.input_dim] }
    pub fn output_dim(&self) -> usize { self.output_dim }

    /// Создаёт out_features нейронов Combiner.
    fn make_neurons(&self, params: &[f32], slice: &ParamSlice) -> Vec<CombinerNeuron> {
        let in_feat = self.input_dim;
        let out_feat = self.output_dim;
        let base = slice.start;
        let mut neurons = Vec::with_capacity(out_feat);
        for out_idx in 0..out_feat {
            let offset = base + out_idx * (2 * in_feat + 1);
            let wa: Vec<f32> = (0..in_feat).map(|j| params[offset + j]).collect();
            let wb: Vec<f32> = (0..in_feat).map(|j| params[offset + in_feat + j]).collect();
            let bias = params[offset + 2 * in_feat];
            neurons.push(CombinerNeuron::new(wa, wb, bias));
        }
        neurons
    }

    /// Прямой проход через нейроны.
    fn forward_with_neurons(
        &self,
        a: &Mat<f32>,
        b: &Mat<f32>,
        params: &[f32],
        slice: &ParamSlice,
    ) -> Mat<f32> {
        let neurons = self.make_neurons(params, slice);
        let batch = a.nrows();
        let out_feat = self.output_dim;
        let mut output = Mat::zeros(batch, out_feat);
        for (j, neuron) in neurons.iter().enumerate() {
            let col = neuron.forward_mat(a, b); // (batch, 1)
            for i in 0..batch {
                output[(i, j)] = col[(i, 0)];
            }
        }
        output
    }

    pub fn forward(
        &self,
        input_a: &DynamicTensor,
        input_b: &DynamicTensor,
        params: &[f32],
        slice: &ParamSlice,
    ) -> (DynamicTensor, DynamicContext) {
        match (input_a, input_b) {
            (DynamicTensor::Dim1(a), DynamicTensor::Dim1(b)) => {
                let a_mat = linalg::tensor2d_to_faer(a);
                let b_mat = linalg::tensor2d_to_faer(b);
                let c_mat = self.forward_with_neurons(&a_mat, &b_mat, params, slice);
                let c_tensor = linalg::faer_to_tensor2d(&c_mat);
                let ctx = DynamicContext::Ctx1D(
                    crate::layers::context1d::LayerContext1D::Combiner {
                        input_a: a.clone(),
                        input_b: b.clone(),
                        pre_act: Vec::new(),
                    },
                );
                (DynamicTensor::Dim1(c_tensor), ctx)
            }
            (DynamicTensor::Dim2(a), DynamicTensor::Dim2(b)) => {
                let a_mat = linalg::tensor3d_to_faer(a);
                let b_mat = linalg::tensor3d_to_faer(b);
                let c_mat = self.forward_with_neurons(&a_mat, &b_mat, params, slice);
                let c_tensor = linalg::faer_to_tensor3d(&c_mat, a.dim1, a.dim2, self.output_dim);
                let ctx = DynamicContext::Ctx2D(
                    crate::layers::context2d::LayerContext::Combiner2D {
                        input_a: a.clone(),
                        input_b: b.clone(),
                        pre_act: Vec::new(),
                    },
                );
                (DynamicTensor::Dim2(c_tensor), ctx)
            }
            (DynamicTensor::Dim3(a), DynamicTensor::Dim3(b)) => {
                let a_mat = linalg::tensor4d_to_faer(a);
                let b_mat = linalg::tensor4d_to_faer(b);
                let c_mat = self.forward_with_neurons(&a_mat, &b_mat, params, slice);
                let c_tensor = linalg::faer_to_tensor4d(&c_mat, a.dim1, a.dim2, a.dim3, self.output_dim);
                let ctx = DynamicContext::Ctx3D(
                    crate::layers::context3d::LayerContext3D::Combiner3D {
                        input_a: a.clone(),
                        input_b: b.clone(),
                        pre_act: Vec::new(),
                    },
                );
                (DynamicTensor::Dim3(c_tensor), ctx)
            }
            (DynamicTensor::Dim4(a), DynamicTensor::Dim4(b)) => {
                let a_mat = linalg::tensor5d_to_faer(a);
                let b_mat = linalg::tensor5d_to_faer(b);
                let c_mat = self.forward_with_neurons(&a_mat, &b_mat, params, slice);
                let c_tensor = linalg::faer_to_tensor5d(&c_mat, a.dim1, a.dim2, a.dim3, a.dim4, self.output_dim);
                let ctx = DynamicContext::Ctx4D(
                    crate::layers::context4d::LayerContext4D::Combiner4D {
                        input_a: a.clone(),
                        input_b: b.clone(),
                        pre_act: Vec::new(),
                    },
                );
                (DynamicTensor::Dim4(c_tensor), ctx)
            }
            _ => panic!("Оба входа должны иметь одинаковую размерность"),
        }
    }

    pub fn backward(
        &self,
        ctx: &DynamicContext,
        delta: &DynamicTensor,
        params: &[f32],
        slice: &ParamSlice,
    ) -> (Vec<DynamicTensor>, Vec<f32>) {
        // Обратный проход оставлен матричным, т.к. он эффективнее.
        match (ctx, delta) {
            (DynamicContext::Ctx1D(c), DynamicTensor::Dim1(d)) => {
                let a_tensor = match &c { crate::layers::context1d::LayerContext1D::Combiner { input_a, .. } => input_a, _ => panic!() };
                let b_tensor = match &c { crate::layers::context1d::LayerContext1D::Combiner { input_b, .. } => input_b, _ => panic!() };
                let a_mat = linalg::tensor2d_to_faer(a_tensor);
                let b_mat = linalg::tensor2d_to_faer(b_tensor);
                let d_mat = linalg::tensor2d_to_faer(d);
                let (da_mat, db_mat, grad) = self.backward_mat(&a_mat, &b_mat, &d_mat, params, slice);
                let da_tensor = linalg::faer_to_tensor2d(&da_mat);
                let db_tensor = linalg::faer_to_tensor2d(&db_mat);
                (vec![DynamicTensor::Dim1(da_tensor), DynamicTensor::Dim1(db_tensor)], grad)
            }
            (DynamicContext::Ctx2D(c), DynamicTensor::Dim2(d)) => {
                let a_tensor = match &c { crate::layers::context2d::LayerContext::Combiner2D { input_a, .. } => input_a, _ => panic!() };
                let b_tensor = match &c { crate::layers::context2d::LayerContext::Combiner2D { input_b, .. } => input_b, _ => panic!() };
                let a_mat = linalg::tensor3d_to_faer(a_tensor);
                let b_mat = linalg::tensor3d_to_faer(b_tensor);
                let d_mat = linalg::tensor3d_to_faer(d);
                let (da_mat, db_mat, grad) = self.backward_mat(&a_mat, &b_mat, &d_mat, params, slice);
                let da_tensor = linalg::faer_to_tensor3d(&da_mat, a_tensor.dim1, a_tensor.dim2, self.input_dim);
                let db_tensor = linalg::faer_to_tensor3d(&db_mat, a_tensor.dim1, a_tensor.dim2, self.input_dim);
                (vec![DynamicTensor::Dim2(da_tensor), DynamicTensor::Dim2(db_tensor)], grad)
            }
            (DynamicContext::Ctx3D(c), DynamicTensor::Dim3(d)) => {
                let a_tensor = match &c { crate::layers::context3d::LayerContext3D::Combiner3D { input_a, .. } => input_a, _ => panic!() };
                let b_tensor = match &c { crate::layers::context3d::LayerContext3D::Combiner3D { input_b, .. } => input_b, _ => panic!() };
                let a_mat = linalg::tensor4d_to_faer(a_tensor);
                let b_mat = linalg::tensor4d_to_faer(b_tensor);
                let d_mat = linalg::tensor4d_to_faer(d);
                let (da_mat, db_mat, grad) = self.backward_mat(&a_mat, &b_mat, &d_mat, params, slice);
                let da_tensor = linalg::faer_to_tensor4d(&da_mat, a_tensor.dim1, a_tensor.dim2, a_tensor.dim3, self.input_dim);
                let db_tensor = linalg::faer_to_tensor4d(&db_mat, a_tensor.dim1, a_tensor.dim2, a_tensor.dim3, self.input_dim);
                (vec![DynamicTensor::Dim3(da_tensor), DynamicTensor::Dim3(db_tensor)], grad)
            }
            (DynamicContext::Ctx4D(c), DynamicTensor::Dim4(d)) => {
                let a_tensor = match &c { crate::layers::context4d::LayerContext4D::Combiner4D { input_a, .. } => input_a, _ => panic!() };
                let b_tensor = match &c { crate::layers::context4d::LayerContext4D::Combiner4D { input_b, .. } => input_b, _ => panic!() };
                let a_mat = linalg::tensor5d_to_faer(a_tensor);
                let b_mat = linalg::tensor5d_to_faer(b_tensor);
                let d_mat = linalg::tensor5d_to_faer(d);
                let (da_mat, db_mat, grad) = self.backward_mat(&a_mat, &b_mat, &d_mat, params, slice);
                let da_tensor = linalg::faer_to_tensor5d(&da_mat, a_tensor.dim1, a_tensor.dim2, a_tensor.dim3, a_tensor.dim4, self.input_dim);
                let db_tensor = linalg::faer_to_tensor5d(&db_mat, a_tensor.dim1, a_tensor.dim2, a_tensor.dim3, a_tensor.dim4, self.input_dim);
                (vec![DynamicTensor::Dim4(da_tensor), DynamicTensor::Dim4(db_tensor)], grad)
            }
            _ => panic!("Несовпадение размерностей контекста и градиента"),
        }
    }

    pub(crate) fn backward_mat(&self, a: &Mat<f32>, b: &Mat<f32>, d_out: &Mat<f32>, params: &[f32], slice: &ParamSlice) -> (Mat<f32>, Mat<f32>, Vec<f32>) {
        let n = self.input_dim;
        let m = self.output_dim;
        let base = slice.start;

        let wa = Mat::from_fn(m, n, |r, c| params[base + r * n + c]);
        let wb = Mat::from_fn(m, n, |r, c| params[base + m * n + r * n + c]);
        let bias_start = base + 2 * m * n;

        let mut pre = a * &wa.transpose() + b * &wb.transpose();
        let rows = a.nrows();
        for r in 0..rows {
            for c in 0..m {
                pre[(r, c)] += params[bias_start + c];
            }
        }

        let d_pre = Mat::from_fn(rows, m, |r, c| {
            if pre[(r, c)] > 0.0 { d_out[(r, c)] } else { 0.0 }
        });

        let da = &d_pre * &wa;
        let db = &d_pre * &wb;

        let d_wa = &d_pre.transpose() * a;
        let d_wb = &d_pre.transpose() * b;
        let mut d_bias = vec![0.0f32; m];
        for r in 0..rows {
            for c in 0..m {
                d_bias[c] += d_pre[(r, c)];
            }
        }

        let mut grad = Vec::with_capacity(self.param_len());
        for r in 0..m {
            for c in 0..n {
                grad.push(d_wa[(r, c)]);
            }
        }
        for r in 0..m {
            for c in 0..n {
                grad.push(d_wb[(r, c)]);
            }
        }
        grad.extend_from_slice(&d_bias);

        (da, db, grad)
    }

    pub fn param_len(&self) -> usize {
        2 * self.output_dim * self.input_dim + self.output_dim
    }
}