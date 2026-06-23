// src/layers/softmax/softmax.rs

use crate::compute_manager::dim_change::DynamicTensor;
use crate::compute_manager::graph::types::DynamicContext;
use crate::model_plan::param_store::ParamSlice;
use crate::layers::UniversalLayer;
use crate::linalg;
use crate::neuron::Softmax as SoftmaxNeuron;
use crate::neuron::base::Neuron;
use faer::Mat;

pub struct Softmax;

impl Softmax {
    pub fn new() -> Self { Self }
}

impl UniversalLayer for Softmax {
    fn forward(
        &self,
        input: &DynamicTensor,
        _params: &[f32],
        _slice: &ParamSlice,
    ) -> (DynamicTensor, DynamicContext) {
        match input {
            DynamicTensor::Dim1(t) => {
                let x_mat = linalg::tensor2d_to_faer(t);
                let y_mat = SoftmaxNeuron.forward_mat(&x_mat);
                let out_tensor = linalg::faer_to_tensor2d(&y_mat);
                let ctx = DynamicContext::Ctx1D(
                    crate::layers::context1d::LayerContext1D::Softmax {
                        output: out_tensor.clone(),
                    },
                );
                (DynamicTensor::Dim1(out_tensor), ctx)
            }
            DynamicTensor::Dim2(t) => {
                let x_mat = linalg::tensor3d_to_faer(t);
                let y_mat = SoftmaxNeuron.forward_mat(&x_mat);
                let out_tensor = linalg::faer_to_tensor3d(&y_mat, t.dim1, t.dim2, t.dim3);
                let ctx = DynamicContext::Ctx2D(
                    crate::layers::context2d::LayerContext::Softmax2D {
                        output: out_tensor.clone(),
                    },
                );
                (DynamicTensor::Dim2(out_tensor), ctx)
            }
            DynamicTensor::Dim3(t) => {
                let x_mat = linalg::tensor4d_to_faer(t);
                let y_mat = SoftmaxNeuron.forward_mat(&x_mat);
                let out_tensor = linalg::faer_to_tensor4d(&y_mat, t.dim1, t.dim2, t.dim3, t.dim4);
                let ctx = DynamicContext::Ctx3D(
                    crate::layers::context3d::LayerContext3D::Softmax3D {
                        output: out_tensor.clone(),
                    },
                );
                (DynamicTensor::Dim3(out_tensor), ctx)
            }
            DynamicTensor::Dim4(t) => {
                let x_mat = linalg::tensor5d_to_faer(t);
                let y_mat = SoftmaxNeuron.forward_mat(&x_mat);
                let out_tensor = linalg::faer_to_tensor5d(&y_mat, t.dim1, t.dim2, t.dim3, t.dim4, t.dim5);
                let ctx = DynamicContext::Ctx4D(
                    crate::layers::context4d::LayerContext4D::Softmax4D {
                        output: out_tensor.clone(),
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
        // Производная softmax: dx = y * (dout - sum(y * dout, по последней оси))
        match (ctx, delta) {
            (DynamicContext::Ctx1D(c), DynamicTensor::Dim1(d)) => {
                let y = match c {
                    crate::layers::context1d::LayerContext1D::Softmax { output } => output,
                    _ => panic!("Expected Softmax context"),
                };
                let y_mat = linalg::tensor2d_to_faer(y);
                let dout_mat = linalg::tensor2d_to_faer(d);
                let dx_mat = softmax_backward_mat(&y_mat, &dout_mat);
                let dx_tensor = linalg::faer_to_tensor2d(&dx_mat);
                (DynamicTensor::Dim1(dx_tensor), vec![])
            }
            (DynamicContext::Ctx2D(c), DynamicTensor::Dim2(d)) => {
                let y = match c {
                    crate::layers::context2d::LayerContext::Softmax2D { output } => output,
                    _ => panic!("Expected Softmax2D context"),
                };
                let y_mat = linalg::tensor3d_to_faer(y);
                let dout_mat = linalg::tensor3d_to_faer(d);
                let dx_mat = softmax_backward_mat(&y_mat, &dout_mat);
                let dx_tensor = linalg::faer_to_tensor3d(&dx_mat, y.dim1, y.dim2, y.dim3);
                (DynamicTensor::Dim2(dx_tensor), vec![])
            }
            (DynamicContext::Ctx3D(c), DynamicTensor::Dim3(d)) => {
                let y = match c {
                    crate::layers::context3d::LayerContext3D::Softmax3D { output } => output,
                    _ => panic!("Expected Softmax3D context"),
                };
                let y_mat = linalg::tensor4d_to_faer(y);
                let dout_mat = linalg::tensor4d_to_faer(d);
                let dx_mat = softmax_backward_mat(&y_mat, &dout_mat);
                let dx_tensor = linalg::faer_to_tensor4d(&dx_mat, y.dim1, y.dim2, y.dim3, y.dim4);
                (DynamicTensor::Dim3(dx_tensor), vec![])
            }
            (DynamicContext::Ctx4D(c), DynamicTensor::Dim4(d)) => {
                let y = match c {
                    crate::layers::context4d::LayerContext4D::Softmax4D { output } => output,
                    _ => panic!("Expected Softmax4D context"),
                };
                let y_mat = linalg::tensor5d_to_faer(y);
                let dout_mat = linalg::tensor5d_to_faer(d);
                let dx_mat = softmax_backward_mat(&y_mat, &dout_mat);
                let dx_tensor = linalg::faer_to_tensor5d(&dx_mat, y.dim1, y.dim2, y.dim3, y.dim4, y.dim5);
                (DynamicTensor::Dim4(dx_tensor), vec![])
            }
            _ => panic!("Mismatched dimensions in Softmax backward"),
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
        match (input, output) {
            (DynamicTensor::Dim1(in_t), DynamicTensor::Dim1(out_t)) => {
                let features = in_t.dim2;
                let x_full = linalg::tensor2d_to_faer(in_t);
                let x_chunk = x_full.submatrix(task_offset, 0, task_count, features);
                let y_chunk = SoftmaxNeuron.forward_mat(&x_chunk.to_owned());
                for (i, r) in (task_offset..task_offset + task_count).enumerate() {
                    for c in 0..features {
                        out_t.data[r][c] = y_chunk[(i, c)];
                    }
                }
            }
            (DynamicTensor::Dim2(in_t), DynamicTensor::Dim2(out_t)) => {
                let dim2 = in_t.dim2;
                let features = in_t.dim3;
                let x_full = linalg::tensor3d_to_faer(in_t);
                let x_chunk = x_full.submatrix(task_offset, 0, task_count, features);
                let y_chunk = SoftmaxNeuron.forward_mat(&x_chunk.to_owned());
                for (idx, flat_idx) in (task_offset..task_offset + task_count).enumerate() {
                    let i = flat_idx / dim2;
                    let j = flat_idx % dim2;
                    let out_row = &mut out_t.data[i][j];
                    for c in 0..features {
                        out_row[c] = y_chunk[(idx, c)];
                    }
                }
            }
            (DynamicTensor::Dim3(in_t), DynamicTensor::Dim3(out_t)) => {
                let dim2 = in_t.dim2;
                let dim3 = in_t.dim3;
                let features = in_t.dim4;
                let x_full = linalg::tensor4d_to_faer(in_t);
                let x_chunk = x_full.submatrix(task_offset, 0, task_count, features);
                let y_chunk = SoftmaxNeuron.forward_mat(&x_chunk.to_owned());
                for (idx, flat_idx) in (task_offset..task_offset + task_count).enumerate() {
                    let i = flat_idx / (dim2 * dim3);
                    let rem = flat_idx % (dim2 * dim3);
                    let j = rem / dim3;
                    let k = rem % dim3;
                    let out_row = &mut out_t.data[i][j][k];
                    for c in 0..features {
                        out_row[c] = y_chunk[(idx, c)];
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
                let y_chunk = SoftmaxNeuron.forward_mat(&x_chunk.to_owned());
                for (idx, flat_idx) in (task_offset..task_offset + task_count).enumerate() {
                    let i = flat_idx / (dim2 * dim3 * dim4);
                    let rem1 = flat_idx % (dim2 * dim3 * dim4);
                    let j = rem1 / (dim3 * dim4);
                    let rem2 = rem1 % (dim3 * dim4);
                    let k = rem2 / dim4;
                    let l = rem2 % dim4;
                    let out_row = &mut out_t.data[i][j][k][l];
                    for c in 0..features {
                        out_row[c] = y_chunk[(idx, c)];
                    }
                }
            }
            _ => panic!("Mismatched tensor dimensions in execute_tasks for Softmax"),
        }
    }

    fn create_sample_context(
        &self,
        _input_sample: &DynamicTensor,
        output_sample: &DynamicTensor,
    ) -> DynamicContext {
        match output_sample {
            DynamicTensor::Dim1(t) => {
                DynamicContext::Ctx1D(crate::layers::context1d::LayerContext1D::Softmax {
                    output: t.clone(),
                })
            }
            DynamicTensor::Dim2(t) => {
                DynamicContext::Ctx2D(crate::layers::context2d::LayerContext::Softmax2D {
                    output: t.clone(),
                })
            }
            DynamicTensor::Dim3(t) => {
                DynamicContext::Ctx3D(crate::layers::context3d::LayerContext3D::Softmax3D {
                    output: t.clone(),
                })
            }
            DynamicTensor::Dim4(t) => {
                DynamicContext::Ctx4D(crate::layers::context4d::LayerContext4D::Softmax4D {
                    output: t.clone(),
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

/// Вспомогательная функция: dx = y * (dout - sum(y * dout)), где сумма берётся по последней оси (поэлементно для каждой строки)
fn softmax_backward_mat(y: &Mat<f32>, dout: &Mat<f32>) -> Mat<f32> {
    let rows = dout.nrows();
    let cols = dout.ncols();
    let mut dx = Mat::zeros(rows, cols);
    for r in 0..rows {
        // Вычисляем dot = sum_j y_{r,j} * dout_{r,j}
        let mut dot = 0.0f32;
        for c in 0..cols {
            dot += y[(r, c)] * dout[(r, c)];
        }
        // Для каждого j: dx = y * (dout - dot)
        for c in 0..cols {
            dx[(r, c)] = y[(r, c)] * (dout[(r, c)] - dot);
        }
    }
    dx
}