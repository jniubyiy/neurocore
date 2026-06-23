// src/layers/relu/relu.rs

use crate::compute_manager::dim_change::DynamicTensor;
use crate::compute_manager::graph::types::DynamicContext;
use crate::model_plan::param_store::ParamSlice;
use crate::layers::UniversalLayer;
use crate::linalg;
use crate::neuron::ReLU as ReLUNeuron;
use crate::neuron::base::Neuron;
use faer::Mat;

pub struct ReLU;

impl ReLU {
    pub fn new() -> Self { Self }
}

impl UniversalLayer for ReLU {
    fn forward(
        &self,
        input: &DynamicTensor,
        _params: &[f32],
        _slice: &ParamSlice,
    ) -> (DynamicTensor, DynamicContext) {
        let relu = ReLUNeuron;

        match input {
            DynamicTensor::Dim1(t) => {
                let x_mat = linalg::tensor2d_to_faer(t); // (batch, features)
                let mut y_mat = Mat::zeros(x_mat.nrows(), x_mat.ncols());
                for j in 0..x_mat.ncols() {
                    let col = x_mat.submatrix(0, j, x_mat.nrows(), 1);
                    let activated = relu.forward_mat(&col.to_owned());
                    for i in 0..x_mat.nrows() {
                        y_mat[(i, j)] = activated[(i, 0)];
                    }
                }
                let out_tensor = linalg::faer_to_tensor2d(&y_mat);
                let ctx = DynamicContext::Ctx1D(
                    crate::layers::context1d::LayerContext1D::ReLU {
                        input: t.clone(),
                    },
                );
                (DynamicTensor::Dim1(out_tensor), ctx)
            }
            DynamicTensor::Dim2(t) => {
                let dim1 = t.dim1;
                let dim2 = t.dim2;
                let features = t.dim3;
                let x_mat = linalg::tensor3d_to_faer(t);
                let mut y_mat = Mat::zeros(x_mat.nrows(), features);
                for j in 0..features {
                    let col = x_mat.submatrix(0, j, x_mat.nrows(), 1);
                    let activated = relu.forward_mat(&col.to_owned());
                    for i in 0..x_mat.nrows() {
                        y_mat[(i, j)] = activated[(i, 0)];
                    }
                }
                let out_tensor = linalg::faer_to_tensor3d(&y_mat, dim1, dim2, features);
                let ctx = DynamicContext::Ctx2D(
                    crate::layers::context2d::LayerContext::ReLU2D {
                        input: t.clone(),
                    },
                );
                (DynamicTensor::Dim2(out_tensor), ctx)
            }
            DynamicTensor::Dim3(t) => {
                let dim1 = t.dim1;
                let dim2 = t.dim2;
                let dim3 = t.dim3;
                let features = t.dim4;
                let x_mat = linalg::tensor4d_to_faer(t);
                let mut y_mat = Mat::zeros(x_mat.nrows(), features);
                for j in 0..features {
                    let col = x_mat.submatrix(0, j, x_mat.nrows(), 1);
                    let activated = relu.forward_mat(&col.to_owned());
                    for i in 0..x_mat.nrows() {
                        y_mat[(i, j)] = activated[(i, 0)];
                    }
                }
                let out_tensor = linalg::faer_to_tensor4d(&y_mat, dim1, dim2, dim3, features);
                let ctx = DynamicContext::Ctx3D(
                    crate::layers::context3d::LayerContext3D::ReLU3D {
                        input: t.clone(),
                    },
                );
                (DynamicTensor::Dim3(out_tensor), ctx)
            }
            DynamicTensor::Dim4(t) => {
                let dim1 = t.dim1;
                let dim2 = t.dim2;
                let dim3 = t.dim3;
                let dim4 = t.dim4;
                let features = t.dim5;
                let x_mat = linalg::tensor5d_to_faer(t);
                let mut y_mat = Mat::zeros(x_mat.nrows(), features);
                for j in 0..features {
                    let col = x_mat.submatrix(0, j, x_mat.nrows(), 1);
                    let activated = relu.forward_mat(&col.to_owned());
                    for i in 0..x_mat.nrows() {
                        y_mat[(i, j)] = activated[(i, 0)];
                    }
                }
                let out_tensor = linalg::faer_to_tensor5d(&y_mat, dim1, dim2, dim3, dim4, features);
                let ctx = DynamicContext::Ctx4D(
                    crate::layers::context4d::LayerContext4D::ReLU4D {
                        input: t.clone(),
                    },
                );
                (DynamicTensor::Dim4(out_tensor), ctx)
            }
        }
    }

    fn backward(
        &self,
        ctx: &DynamicContext,
        delta: &DynamicTensor,
        _params: &[f32],
        _slice: &ParamSlice,
    ) -> (DynamicTensor, Vec<f32>) {
        // ReLU backward: dx = delta * (x > 0)
        match (ctx, delta) {
            (DynamicContext::Ctx1D(c), DynamicTensor::Dim1(d)) => {
                let x = match c {
                    crate::layers::context1d::LayerContext1D::ReLU { input } => input,
                    _ => panic!("Expected ReLU context"),
                };
                let x_mat = linalg::tensor2d_to_faer(x);
                let d_mat = linalg::tensor2d_to_faer(d);
                let dx_mat = relu_backward_mat(&x_mat, &d_mat);
                let dx_tensor = linalg::faer_to_tensor2d(&dx_mat);
                (DynamicTensor::Dim1(dx_tensor), vec![])
            }
            (DynamicContext::Ctx2D(c), DynamicTensor::Dim2(d)) => {
                let x = match c {
                    crate::layers::context2d::LayerContext::ReLU2D { input } => input,
                    _ => panic!("Expected ReLU2D context"),
                };
                let x_mat = linalg::tensor3d_to_faer(x);
                let d_mat = linalg::tensor3d_to_faer(d);
                let dx_mat = relu_backward_mat(&x_mat, &d_mat);
                let dx_tensor = linalg::faer_to_tensor3d(&dx_mat, x.dim1, x.dim2, x.dim3);
                (DynamicTensor::Dim2(dx_tensor), vec![])
            }
            (DynamicContext::Ctx3D(c), DynamicTensor::Dim3(d)) => {
                let x = match c {
                    crate::layers::context3d::LayerContext3D::ReLU3D { input } => input,
                    _ => panic!("Expected ReLU3D context"),
                };
                let x_mat = linalg::tensor4d_to_faer(x);
                let d_mat = linalg::tensor4d_to_faer(d);
                let dx_mat = relu_backward_mat(&x_mat, &d_mat);
                let dx_tensor = linalg::faer_to_tensor4d(&dx_mat, x.dim1, x.dim2, x.dim3, x.dim4);
                (DynamicTensor::Dim3(dx_tensor), vec![])
            }
            (DynamicContext::Ctx4D(c), DynamicTensor::Dim4(d)) => {
                let x = match c {
                    crate::layers::context4d::LayerContext4D::ReLU4D { input } => input,
                    _ => panic!("Expected ReLU4D context"),
                };
                let x_mat = linalg::tensor5d_to_faer(x);
                let d_mat = linalg::tensor5d_to_faer(d);
                let dx_mat = relu_backward_mat(&x_mat, &d_mat);
                let dx_tensor = linalg::faer_to_tensor5d(&dx_mat, x.dim1, x.dim2, x.dim3, x.dim4, x.dim5);
                (DynamicTensor::Dim4(dx_tensor), vec![])
            }
            _ => panic!("Mismatched dimensions in ReLU backward"),
        }
    }

    fn param_len(&self) -> usize { 0 }
    fn input_features(&self) -> usize { 0 }
    fn output_features(&self) -> usize { 0 }

    fn total_tasks(&self, input: &DynamicTensor) -> usize {
        input.batch_size()
    }

    fn execute_tasks(
        &self,
        input: &DynamicTensor,
        output: &mut DynamicTensor,
        task_offset: usize,
        task_count: usize,
        _params: &[f32],
        _slice: &ParamSlice,
    ) {
        let relu = ReLUNeuron;
        match (input, output) {
            (DynamicTensor::Dim1(in_t), DynamicTensor::Dim1(out_t)) => {
                let features = in_t.dim2;
                let x_full = linalg::tensor2d_to_faer(in_t);
                let x_chunk = x_full.submatrix(task_offset, 0, task_count, features);
                for j in 0..features {
                    let col = x_chunk.submatrix(0, j, task_count, 1);
                    let activated = relu.forward_mat(&col.to_owned());
                    for (idx, i) in (task_offset..task_offset + task_count).enumerate() {
                        out_t.data[i][j] = activated[(idx, 0)];
                    }
                }
            }
            (DynamicTensor::Dim2(in_t), DynamicTensor::Dim2(out_t)) => {
                let dim2 = in_t.dim2;
                let features = in_t.dim3;
                let x_full = linalg::tensor3d_to_faer(in_t);
                let x_chunk = x_full.submatrix(task_offset, 0, task_count, features);
                for j in 0..features {
                    let col = x_chunk.submatrix(0, j, task_count, 1);
                    let activated = relu.forward_mat(&col.to_owned());
                    for (idx, flat_idx) in (task_offset..task_offset + task_count).enumerate() {
                        let i = flat_idx / dim2;
                        let jj = flat_idx % dim2;
                        out_t.data[i][jj][j] = activated[(idx, 0)];
                    }
                }
            }
            (DynamicTensor::Dim3(in_t), DynamicTensor::Dim3(out_t)) => {
                let dim2 = in_t.dim2;
                let dim3 = in_t.dim3;
                let features = in_t.dim4;
                let x_full = linalg::tensor4d_to_faer(in_t);
                let x_chunk = x_full.submatrix(task_offset, 0, task_count, features);
                for j in 0..features {
                    let col = x_chunk.submatrix(0, j, task_count, 1);
                    let activated = relu.forward_mat(&col.to_owned());
                    for (idx, flat_idx) in (task_offset..task_offset + task_count).enumerate() {
                        let i = flat_idx / (dim2 * dim3);
                        let rem = flat_idx % (dim2 * dim3);
                        let jj = rem / dim3;
                        let k = rem % dim3;
                        out_t.data[i][jj][k][j] = activated[(idx, 0)];
                    }
                }
            }
            (DynamicTensor::Dim4(in_t), DynamicTensor::Dim4(out_t)) => {
                let dim2 = in_t.dim2;
                let dim3 = in_t.dim3;
                let dim4 = in_t.dim4;
                let features = in_t.dim5;
                let x_full = linalg::tensor5d_to_faer(in_t);
                let x_chunk = x_full.submatrix(task_offset, 0, task_count, features);
                for j in 0..features {
                    let col = x_chunk.submatrix(0, j, task_count, 1);
                    let activated = relu.forward_mat(&col.to_owned());
                    for (idx, flat_idx) in (task_offset..task_offset + task_count).enumerate() {
                        let i = flat_idx / (dim2 * dim3 * dim4);
                        let rem1 = flat_idx % (dim2 * dim3 * dim4);
                        let jj = rem1 / (dim3 * dim4);
                        let rem2 = rem1 % (dim3 * dim4);
                        let k = rem2 / dim4;
                        let l = rem2 % dim4;
                        out_t.data[i][jj][k][l][j] = activated[(idx, 0)];
                    }
                }
            }
            _ => panic!("Mismatched tensor dimensions in execute_tasks for ReLU"),
        }
    }

    fn create_sample_context(
        &self,
        input_sample: &DynamicTensor,
        _output_sample: &DynamicTensor,
    ) -> DynamicContext {
        match input_sample {
            DynamicTensor::Dim1(t) => {
                DynamicContext::Ctx1D(crate::layers::context1d::LayerContext1D::ReLU {
                    input: t.clone(),
                })
            }
            DynamicTensor::Dim2(t) => {
                DynamicContext::Ctx2D(crate::layers::context2d::LayerContext::ReLU2D {
                    input: t.clone(),
                })
            }
            DynamicTensor::Dim3(t) => {
                DynamicContext::Ctx3D(crate::layers::context3d::LayerContext3D::ReLU3D {
                    input: t.clone(),
                })
            }
            DynamicTensor::Dim4(t) => {
                DynamicContext::Ctx4D(crate::layers::context4d::LayerContext4D::ReLU4D {
                    input: t.clone(),
                })
            }
        }
    }

    fn output_tensor_shape(&self, input: &DynamicTensor) -> DynamicTensor {
        match input {
            DynamicTensor::Dim1(t) => DynamicTensor::Dim1(crate::tensor::Tensor2D::zeros(t.dim1, t.dim2)),
            DynamicTensor::Dim2(t) => DynamicTensor::Dim2(crate::tensor::Tensor3D::zeros(t.dim1, t.dim2, t.dim3)),
            DynamicTensor::Dim3(t) => DynamicTensor::Dim3(crate::tensor::Tensor4D::zeros(t.dim1, t.dim2, t.dim3, t.dim4)),
            DynamicTensor::Dim4(t) => DynamicTensor::Dim4(crate::tensor::Tensor5D::zeros(t.dim1, t.dim2, t.dim3, t.dim4, t.dim5)),
        }
    }
}

/// Вспомогательная функция для вычисления градиента ReLU: dx = dout * (x > 0)
fn relu_backward_mat(x: &Mat<f32>, dout: &Mat<f32>) -> Mat<f32> {
    let rows = dout.nrows();
    let cols = dout.ncols();
    let mut dx = Mat::zeros(rows, cols);
    for r in 0..rows {
        for c in 0..cols {
            if x[(r, c)] > 0.0 {
                dx[(r, c)] = dout[(r, c)];
            }
        }
    }
    dx
}