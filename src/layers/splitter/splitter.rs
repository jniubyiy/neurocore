// src/layers/splitter/splitter.rs

use crate::compute_manager::dim_change::DynamicTensor;
use crate::compute_manager::graph::types::DynamicContext;
use crate::model_plan::param_store::ParamSlice;
use crate::linalg;
use faer::Mat;
use crate::neuron::Splitter as SplitterNeuron;

pub struct Splitter {
    input_dim: usize,
    output_dims: Vec<usize>,
}

impl Splitter {
    pub fn new(input_dim: usize, output_dims: Vec<usize>) -> Self {
        assert_eq!(output_dims.len(), 2, "Splitter поддерживает ровно два выхода");
        Self { input_dim, output_dims }
    }

    pub fn input_dim(&self) -> usize { self.input_dim }
    pub fn output_dims(&self) -> &[usize] { &self.output_dims }

    fn make_neuron(&self, params: &[f32], slice: &ParamSlice) -> SplitterNeuron {
        let n = self.input_dim;
        let p = self.output_dims[0];
        let q = self.output_dims[1];
        let base = slice.start;
        let wa_len = p * n;
        let wb_len = q * n;
        let ba_len = p;
        let bb_len = q;
        let wa = params[base..base + wa_len].to_vec();
        let wb = params[base + wa_len..base + wa_len + wb_len].to_vec();
        let bias_a = params[base + wa_len + wb_len..base + wa_len + wb_len + ba_len].to_vec();
        let bias_b = params[base + wa_len + wb_len + ba_len..base + wa_len + wb_len + ba_len + bb_len].to_vec();

        let mut neuron = SplitterNeuron::new(n, p, q);
        let mut flat = Vec::with_capacity(neuron.param_count());
        flat.extend_from_slice(&wa);
        flat.extend_from_slice(&wb);
        flat.extend_from_slice(&bias_a);
        flat.extend_from_slice(&bias_b);
        neuron.set_params(&flat);
        neuron
    }

    pub fn forward(
        &self,
        input: &DynamicTensor,
        params: &[f32],
        slice: &ParamSlice,
    ) -> (Vec<DynamicTensor>, DynamicContext) {
        let neuron = self.make_neuron(params, slice);
        let p = self.output_dims[0];
        let q = self.output_dims[1];

        match input {
            DynamicTensor::Dim1(t) => {
                let x_mat = linalg::tensor2d_to_faer(t);
                let (a_mat, b_mat, pre_a_mat, pre_b_mat) = neuron.forward_mat(&x_mat);
                let a_tensor = linalg::faer_to_tensor2d(&a_mat);
                let b_tensor = linalg::faer_to_tensor2d(&b_mat);
                let pre_a_flat: Vec<f32> = mat_to_flat(&pre_a_mat);
                let pre_b_flat: Vec<f32> = mat_to_flat(&pre_b_mat);
                let ctx = DynamicContext::Ctx1D(
                    crate::layers::context1d::LayerContext1D::Splitter {
                        input: t.clone(),
                        pre_a: pre_a_flat,
                        pre_b: pre_b_flat,
                    },
                );
                (vec![DynamicTensor::Dim1(a_tensor), DynamicTensor::Dim1(b_tensor)], ctx)
            }
            DynamicTensor::Dim2(t) => {
                let dim1 = t.dim1;
                let dim2 = t.dim2;
                let x_mat = linalg::tensor3d_to_faer(t);
                let (a_mat, b_mat, pre_a_mat, pre_b_mat) = neuron.forward_mat(&x_mat);
                let a_tensor = linalg::faer_to_tensor3d(&a_mat, dim1, dim2, p);
                let b_tensor = linalg::faer_to_tensor3d(&b_mat, dim1, dim2, q);
                let ctx = DynamicContext::Ctx2D(
                    crate::layers::context2d::LayerContext::Splitter2D {
                        input: t.clone(),
                        pre_a: mat_to_flat(&pre_a_mat),
                        pre_b: mat_to_flat(&pre_b_mat),
                    },
                );
                (vec![DynamicTensor::Dim2(a_tensor), DynamicTensor::Dim2(b_tensor)], ctx)
            }
            DynamicTensor::Dim3(t) => {
                let dim1 = t.dim1;
                let dim2 = t.dim2;
                let dim3 = t.dim3;
                let x_mat = linalg::tensor4d_to_faer(t);
                let (a_mat, b_mat, pre_a_mat, pre_b_mat) = neuron.forward_mat(&x_mat);
                let a_tensor = linalg::faer_to_tensor4d(&a_mat, dim1, dim2, dim3, p);
                let b_tensor = linalg::faer_to_tensor4d(&b_mat, dim1, dim2, dim3, q);
                let ctx = DynamicContext::Ctx3D(
                    crate::layers::context3d::LayerContext3D::Splitter3D {
                        input: t.clone(),
                        pre_a: mat_to_flat(&pre_a_mat),
                        pre_b: mat_to_flat(&pre_b_mat),
                    },
                );
                (vec![DynamicTensor::Dim3(a_tensor), DynamicTensor::Dim3(b_tensor)], ctx)
            }
            DynamicTensor::Dim4(t) => {
                let dim1 = t.dim1;
                let dim2 = t.dim2;
                let dim3 = t.dim3;
                let dim4 = t.dim4;
                let x_mat = linalg::tensor5d_to_faer(t);
                let (a_mat, b_mat, pre_a_mat, pre_b_mat) = neuron.forward_mat(&x_mat);
                let a_tensor = linalg::faer_to_tensor5d(&a_mat, dim1, dim2, dim3, dim4, p);
                let b_tensor = linalg::faer_to_tensor5d(&b_mat, dim1, dim2, dim3, dim4, q);
                let ctx = DynamicContext::Ctx4D(
                    crate::layers::context4d::LayerContext4D::Splitter4D {
                        input: t.clone(),
                        pre_a: mat_to_flat(&pre_a_mat),
                        pre_b: mat_to_flat(&pre_b_mat),
                    },
                );
                (vec![DynamicTensor::Dim4(a_tensor), DynamicTensor::Dim4(b_tensor)], ctx)
            }
        }
    }

    pub fn backward(
        &self,
        ctx: &DynamicContext,
        deltas: &[DynamicTensor],
        params: &[f32],
        slice: &ParamSlice,
    ) -> (DynamicTensor, Vec<f32>) {
        let neuron = self.make_neuron(params, slice);
        let p = self.output_dims[0];
        let q = self.output_dims[1];

        match ctx {
            DynamicContext::Ctx1D(c) => {
                let x_tensor = match c {
                    crate::layers::context1d::LayerContext1D::Splitter { input, .. } => input,
                    _ => panic!("Expected Splitter context"),
                };
                let a_tensor = match &deltas[0] { DynamicTensor::Dim1(t) => t, _ => panic!() };
                let b_tensor = match &deltas[1] { DynamicTensor::Dim1(t) => t, _ => panic!() };
                let pre_a_flat = match c { crate::layers::context1d::LayerContext1D::Splitter { pre_a, .. } => pre_a.clone(), _ => panic!() };
                let pre_b_flat = match c { crate::layers::context1d::LayerContext1D::Splitter { pre_b, .. } => pre_b.clone(), _ => panic!() };

                let x_mat = linalg::tensor2d_to_faer(x_tensor);
                let da_mat = linalg::tensor2d_to_faer(a_tensor);
                let db_mat = linalg::tensor2d_to_faer(b_tensor);
                let batch = x_mat.nrows();
                let pre_a_mat = flat_to_mat(pre_a_flat, batch, p);
                let pre_b_mat = flat_to_mat(pre_b_flat, batch, q);
                let (dx_mat, grad) = neuron.backward_mat(&x_mat, &da_mat, &db_mat, &pre_a_mat, &pre_b_mat);
                let dx_tensor = linalg::faer_to_tensor2d(&dx_mat);
                (DynamicTensor::Dim1(dx_tensor), grad)
            }
            DynamicContext::Ctx2D(c) => {
                let x_tensor = match c { crate::layers::context2d::LayerContext::Splitter2D { input, .. } => input, _ => panic!() };
                let a_tensor = match &deltas[0] { DynamicTensor::Dim2(t) => t, _ => panic!() };
                let b_tensor = match &deltas[1] { DynamicTensor::Dim2(t) => t, _ => panic!() };
                let pre_a_flat = match c { crate::layers::context2d::LayerContext::Splitter2D { pre_a, .. } => pre_a.clone(), _ => panic!() };
                let pre_b_flat = match c { crate::layers::context2d::LayerContext::Splitter2D { pre_b, .. } => pre_b.clone(), _ => panic!() };

                let dim1 = x_tensor.dim1;
                let dim2 = x_tensor.dim2;
                let total = dim1 * dim2;
                let x_mat = linalg::tensor3d_to_faer(x_tensor);
                let da_mat = linalg::tensor3d_to_faer(a_tensor);
                let db_mat = linalg::tensor3d_to_faer(b_tensor);
                let pre_a_mat = flat_to_mat(pre_a_flat, total, p);
                let pre_b_mat = flat_to_mat(pre_b_flat, total, q);
                let (dx_mat, grad) = neuron.backward_mat(&x_mat, &da_mat, &db_mat, &pre_a_mat, &pre_b_mat);
                let dx_tensor = linalg::faer_to_tensor3d(&dx_mat, dim1, dim2, self.input_dim);
                (DynamicTensor::Dim2(dx_tensor), grad)
            }
            DynamicContext::Ctx3D(c) => {
                let x_tensor = match c { crate::layers::context3d::LayerContext3D::Splitter3D { input, .. } => input, _ => panic!() };
                let a_tensor = match &deltas[0] { DynamicTensor::Dim3(t) => t, _ => panic!() };
                let b_tensor = match &deltas[1] { DynamicTensor::Dim3(t) => t, _ => panic!() };
                let pre_a_flat = match c { crate::layers::context3d::LayerContext3D::Splitter3D { pre_a, .. } => pre_a.clone(), _ => panic!() };
                let pre_b_flat = match c { crate::layers::context3d::LayerContext3D::Splitter3D { pre_b, .. } => pre_b.clone(), _ => panic!() };

                let dim1 = x_tensor.dim1;
                let dim2 = x_tensor.dim2;
                let dim3 = x_tensor.dim3;
                let total = dim1 * dim2 * dim3;
                let x_mat = linalg::tensor4d_to_faer(x_tensor);
                let da_mat = linalg::tensor4d_to_faer(a_tensor);
                let db_mat = linalg::tensor4d_to_faer(b_tensor);
                let pre_a_mat = flat_to_mat(pre_a_flat, total, p);
                let pre_b_mat = flat_to_mat(pre_b_flat, total, q);
                let (dx_mat, grad) = neuron.backward_mat(&x_mat, &da_mat, &db_mat, &pre_a_mat, &pre_b_mat);
                let dx_tensor = linalg::faer_to_tensor4d(&dx_mat, dim1, dim2, dim3, self.input_dim);
                (DynamicTensor::Dim3(dx_tensor), grad)
            }
            DynamicContext::Ctx4D(c) => {
                let x_tensor = match c { crate::layers::context4d::LayerContext4D::Splitter4D { input, .. } => input, _ => panic!() };
                let a_tensor = match &deltas[0] { DynamicTensor::Dim4(t) => t, _ => panic!() };
                let b_tensor = match &deltas[1] { DynamicTensor::Dim4(t) => t, _ => panic!() };
                let pre_a_flat = match c { crate::layers::context4d::LayerContext4D::Splitter4D { pre_a, .. } => pre_a.clone(), _ => panic!() };
                let pre_b_flat = match c { crate::layers::context4d::LayerContext4D::Splitter4D { pre_b, .. } => pre_b.clone(), _ => panic!() };

                let dim1 = x_tensor.dim1;
                let dim2 = x_tensor.dim2;
                let dim3 = x_tensor.dim3;
                let dim4 = x_tensor.dim4;
                let total = dim1 * dim2 * dim3 * dim4;
                let x_mat = linalg::tensor5d_to_faer(x_tensor);
                let da_mat = linalg::tensor5d_to_faer(a_tensor);
                let db_mat = linalg::tensor5d_to_faer(b_tensor);
                let pre_a_mat = flat_to_mat(pre_a_flat, total, p);
                let pre_b_mat = flat_to_mat(pre_b_flat, total, q);
                let (dx_mat, grad) = neuron.backward_mat(&x_mat, &da_mat, &db_mat, &pre_a_mat, &pre_b_mat);
                let dx_tensor = linalg::faer_to_tensor5d(&dx_mat, dim1, dim2, dim3, dim4, self.input_dim);
                (DynamicTensor::Dim4(dx_tensor), grad)
            }
        }
    }

    pub fn param_len(&self) -> usize {
        let p = self.output_dims[0];
        let q = self.output_dims[1];
        self.input_dim * p + self.input_dim * q + p + q
    }
}

fn mat_to_flat(mat: &Mat<f32>) -> Vec<f32> {
    let rows = mat.nrows();
    let cols = mat.ncols();
    let mut flat = Vec::with_capacity(rows * cols);
    for i in 0..rows {
        for j in 0..cols {
            flat.push(mat[(i, j)]);
        }
    }
    flat
}

fn flat_to_mat(flat: Vec<f32>, rows: usize, cols: usize) -> Mat<f32> {
    Mat::from_fn(rows, cols, |i, j| flat[i * cols + j])
}