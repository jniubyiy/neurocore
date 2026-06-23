// src/layers/memory/memory.rs

use crate::compute_manager::dim_change::DynamicTensor;
use crate::compute_manager::graph::types::DynamicContext;
use crate::model_plan::param_store::ParamSlice;
use crate::layers::UniversalLayer;
use crate::linalg;
use faer::Mat;
use crate::neuron::base::Neuron;
use crate::neuron::Memory as MemoryNeuron;

pub struct Memory {
    in_features: usize,
    out_features: usize,
}

impl Memory {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        Self { in_features, out_features }
    }

    /// Создаёт out_features нейронов Memory, извлекая их параметры из среза.
    fn make_neurons(&self, params: &[f32], slice: &ParamSlice) -> Vec<MemoryNeuron> {
        let in_feat = self.in_features;
        let out_feat = self.out_features;
        let base = slice.start;
        let mut neurons = Vec::with_capacity(out_feat);
        for out_idx in 0..out_feat {
            let offset = base + out_idx * (2 * in_feat + 1);
            let w0: Vec<f32> = (0..in_feat).map(|j| params[offset + j]).collect();
            let w1: Vec<f32> = (0..in_feat).map(|j| params[offset + in_feat + j]).collect();
            let temp = params[offset + 2 * in_feat];
            neurons.push(MemoryNeuron::new(w0, w1, temp));
        }
        neurons
    }

    /// Прямой проход с использованием нейронов.
    fn forward_with_neurons(
        &self,
        input: &Mat<f32>,
        params: &[f32],
        slice: &ParamSlice,
    ) -> Mat<f32> {
        let batch = input.nrows();
        let out_feat = self.out_features;
        let neurons = self.make_neurons(params, slice);
        let mut output = Mat::zeros(batch, out_feat);
        for (j, neuron) in neurons.iter().enumerate() {
            let col = neuron.forward_mat(input); // (batch, 1)
            for i in 0..batch {
                output[(i, j)] = col[(i, 0)];
            }
        }
        output
    }

    /// Обратный проход (остаётся внутри слоя)
    fn backward_mat(
        &self,
        x: &Mat<f32>,
        dout: &Mat<f32>,
        params: &[f32],
        slice: &ParamSlice,
    ) -> (Mat<f32>, Vec<f32>) {
        let in_features = self.in_features;
        let out_features = self.out_features;
        let base = slice.start;

        let m0 = Mat::from_fn(out_features, in_features, |r, c| {
            params[base + r * (2 * in_features + 1) + c]
        });
        let m1 = Mat::from_fn(out_features, in_features, |r, c| {
            params[base + r * (2 * in_features + 1) + in_features + c]
        });
        let temps: Vec<f32> = (0..out_features)
            .map(|r| params[base + r * (2 * in_features + 1) + 2 * in_features])
            .collect();

        let batch = x.nrows();
        let dot0 = x * &m0.transpose();
        let dot1 = x * &m1.transpose();

        let mut dx = Mat::zeros(batch, in_features);
        let mut d_m0 = Mat::zeros(out_features, in_features);
        let mut d_m1 = Mat::zeros(out_features, in_features);
        let mut d_t = vec![0.0f32; out_features];

        for n in 0..batch {
            for c in 0..out_features {
                let a = dot0[(n, c)];
                let b = dot1[(n, c)];
                let t = temps[c];
                let z = (a - b) / t;
                let s = 1.0 / (1.0 + (-z).exp());
                let dout_val = dout[(n, c)];

                let ds_dz = s * (1.0 - s);
                let da = dout_val * (s + (a - b) * ds_dz / t);
                let db = dout_val * (1.0 - s - (a - b) * ds_dz / t);

                for i in 0..in_features {
                    dx[(n, i)] += da * m0[(c, i)] + db * m1[(c, i)];
                }

                for i in 0..in_features {
                    d_m0[(c, i)] += da * x[(n, i)];
                    d_m1[(c, i)] += db * x[(n, i)];
                }

                let dz_dt = -(a - b) / (t * t);
                let ds_dt = ds_dz * dz_dt;
                let d_s = dout_val * (a - b);
                d_t[c] += d_s * ds_dt;
            }
        }

        let param_len = self.param_len();
        let mut grad = Vec::with_capacity(param_len);
        for r in 0..out_features {
            for c in 0..in_features {
                grad.push(d_m0[(r, c)]);
            }
        }
        for r in 0..out_features {
            for c in 0..in_features {
                grad.push(d_m1[(r, c)]);
            }
        }
        grad.extend_from_slice(&d_t);

        (dx, grad)
    }
}

impl UniversalLayer for Memory {
    fn forward(
        &self,
        input: &DynamicTensor,
        params: &[f32],
        slice: &ParamSlice,
    ) -> (DynamicTensor, DynamicContext) {
        match input {
            DynamicTensor::Dim1(t) => {
                let x_mat = linalg::tensor2d_to_faer(t);
                let y_mat = self.forward_with_neurons(&x_mat, params, slice);
                let out_tensor = linalg::faer_to_tensor2d(&y_mat);
                let ctx = DynamicContext::Ctx1D(
                    crate::layers::context1d::LayerContext1D::Memory { input: t.clone() },
                );
                (DynamicTensor::Dim1(out_tensor), ctx)
            }
            DynamicTensor::Dim2(t) => {
                let dim1 = t.dim1;
                let dim2 = t.dim2;
                let x_mat = linalg::tensor3d_to_faer(t);
                let y_mat = self.forward_with_neurons(&x_mat, params, slice);
                let out_tensor = linalg::faer_to_tensor3d(&y_mat, dim1, dim2, self.out_features);
                let ctx = DynamicContext::Ctx2D(
                    crate::layers::context2d::LayerContext::Memory2D { input: t.clone() },
                );
                (DynamicTensor::Dim2(out_tensor), ctx)
            }
            DynamicTensor::Dim3(t) => {
                let dim1 = t.dim1;
                let dim2 = t.dim2;
                let dim3 = t.dim3;
                let x_mat = linalg::tensor4d_to_faer(t);
                let y_mat = self.forward_with_neurons(&x_mat, params, slice);
                let out_tensor = linalg::faer_to_tensor4d(&y_mat, dim1, dim2, dim3, self.out_features);
                let ctx = DynamicContext::Ctx3D(
                    crate::layers::context3d::LayerContext3D::Memory3D { input: t.clone() },
                );
                (DynamicTensor::Dim3(out_tensor), ctx)
            }
            DynamicTensor::Dim4(t) => {
                let dim1 = t.dim1;
                let dim2 = t.dim2;
                let dim3 = t.dim3;
                let dim4 = t.dim4;
                let x_mat = linalg::tensor5d_to_faer(t);
                let y_mat = self.forward_with_neurons(&x_mat, params, slice);
                let out_tensor = linalg::faer_to_tensor5d(&y_mat, dim1, dim2, dim3, dim4, self.out_features);
                let ctx = DynamicContext::Ctx4D(
                    crate::layers::context4d::LayerContext4D::Memory4D { input: t.clone() },
                );
                (DynamicTensor::Dim4(out_tensor), ctx)
            }
        }
    }

    fn backward(
        &self,
        ctx: &DynamicContext,
        delta: &DynamicTensor,
        params: &[f32],
        slice: &ParamSlice,
    ) -> (DynamicTensor, Vec<f32>) {
        match (ctx, delta) {
            (DynamicContext::Ctx1D(c), DynamicTensor::Dim1(d)) => {
                let x_tensor = match c {
                    crate::layers::context1d::LayerContext1D::Memory { input } => input,
                    _ => panic!("Expected Memory context"),
                };
                let x_mat = linalg::tensor2d_to_faer(x_tensor);
                let dout_mat = linalg::tensor2d_to_faer(d);
                let (dx_mat, grad) = self.backward_mat(&x_mat, &dout_mat, params, slice);
                let dx_tensor = linalg::faer_to_tensor2d(&dx_mat);
                (DynamicTensor::Dim1(dx_tensor), grad)
            }
            (DynamicContext::Ctx2D(c), DynamicTensor::Dim2(d)) => {
                let x_tensor = match c {
                    crate::layers::context2d::LayerContext::Memory2D { input } => input,
                    _ => panic!("Expected Memory2D context"),
                };
                let x_mat = linalg::tensor3d_to_faer(x_tensor);
                let dout_mat = linalg::tensor3d_to_faer(d);
                let (dx_mat, grad) = self.backward_mat(&x_mat, &dout_mat, params, slice);
                let dx_tensor = linalg::faer_to_tensor3d(&dx_mat, x_tensor.dim1, x_tensor.dim2, self.in_features);
                (DynamicTensor::Dim2(dx_tensor), grad)
            }
            (DynamicContext::Ctx3D(c), DynamicTensor::Dim3(d)) => {
                let x_tensor = match c {
                    crate::layers::context3d::LayerContext3D::Memory3D { input } => input,
                    _ => panic!("Expected Memory3D context"),
                };
                let x_mat = linalg::tensor4d_to_faer(x_tensor);
                let dout_mat = linalg::tensor4d_to_faer(d);
                let (dx_mat, grad) = self.backward_mat(&x_mat, &dout_mat, params, slice);
                let dx_tensor = linalg::faer_to_tensor4d(&dx_mat, x_tensor.dim1, x_tensor.dim2, x_tensor.dim3, self.in_features);
                (DynamicTensor::Dim3(dx_tensor), grad)
            }
            (DynamicContext::Ctx4D(c), DynamicTensor::Dim4(d)) => {
                let x_tensor = match c {
                    crate::layers::context4d::LayerContext4D::Memory4D { input } => input,
                    _ => panic!("Expected Memory4D context"),
                };
                let x_mat = linalg::tensor5d_to_faer(x_tensor);
                let dout_mat = linalg::tensor5d_to_faer(d);
                let (dx_mat, grad) = self.backward_mat(&x_mat, &dout_mat, params, slice);
                let dx_tensor = linalg::faer_to_tensor5d(&dx_mat, x_tensor.dim1, x_tensor.dim2, x_tensor.dim3, x_tensor.dim4, self.in_features);
                (DynamicTensor::Dim4(dx_tensor), grad)
            }
            _ => panic!("Mismatched dimensions in Memory backward"),
        }
    }

    fn param_len(&self) -> usize {
        self.out_features * (2 * self.in_features + 1)
    }
    fn input_features(&self) -> usize { self.in_features }
    fn output_features(&self) -> usize { self.out_features }

    fn total_tasks(&self, input: &DynamicTensor) -> usize {
        input.batch_size()
    }

    fn execute_tasks(
        &self,
        input: &DynamicTensor,
        output: &mut DynamicTensor,
        task_offset: usize,
        task_count: usize,
        params: &[f32],
        slice: &ParamSlice,
    ) {
        let neurons = self.make_neurons(params, slice);
        let in_feat = self.in_features;

        match (input, output) {
            (DynamicTensor::Dim1(in_t), DynamicTensor::Dim1(out_t)) => {
                let x_full = linalg::tensor2d_to_faer(in_t);
                let x_chunk = x_full.submatrix(task_offset, 0, task_count, in_feat);
                for (i, r) in (task_offset..task_offset + task_count).enumerate() {
                    for (j, neuron) in neurons.iter().enumerate() {
                        let row_mat = x_chunk.submatrix(i, 0, 1, in_feat).to_owned();
                        let val = neuron.forward_mat(&row_mat);
                        out_t.data[r][j] = val[(0, 0)];
                    }
                }
            }
            (DynamicTensor::Dim2(in_t), DynamicTensor::Dim2(out_t)) => {
                let dim2 = in_t.dim2;
                let x_full = linalg::tensor3d_to_faer(in_t);
                let x_chunk = x_full.submatrix(task_offset, 0, task_count, in_feat);
                for (idx, flat_idx) in (task_offset..task_offset + task_count).enumerate() {
                    let i = flat_idx / dim2;
                    let j = flat_idx % dim2;
                    let row_mat = x_chunk.submatrix(idx, 0, 1, in_feat).to_owned();
                    for (k, neuron) in neurons.iter().enumerate() {
                        let val = neuron.forward_mat(&row_mat);
                        out_t.data[i][j][k] = val[(0, 0)];
                    }
                }
            }
            (DynamicTensor::Dim3(in_t), DynamicTensor::Dim3(out_t)) => {
                let dim2 = in_t.dim2;
                let dim3 = in_t.dim3;
                let x_full = linalg::tensor4d_to_faer(in_t);
                let x_chunk = x_full.submatrix(task_offset, 0, task_count, in_feat);
                for (idx, flat_idx) in (task_offset..task_offset + task_count).enumerate() {
                    let i = flat_idx / (dim2 * dim3);
                    let rem = flat_idx % (dim2 * dim3);
                    let j = rem / dim3;
                    let k = rem % dim3;
                    let row_mat = x_chunk.submatrix(idx, 0, 1, in_feat).to_owned();
                    for (l, neuron) in neurons.iter().enumerate() {
                        let val = neuron.forward_mat(&row_mat);
                        out_t.data[i][j][k][l] = val[(0, 0)];
                    }
                }
            }
            (DynamicTensor::Dim4(in_t), DynamicTensor::Dim4(out_t)) => {
                let dim2 = in_t.dim2;
                let dim3 = in_t.dim3;
                let dim4 = in_t.dim4;
                let x_full = linalg::tensor5d_to_faer(in_t);
                let x_chunk = x_full.submatrix(task_offset, 0, task_count, in_feat);
                for (idx, flat_idx) in (task_offset..task_offset + task_count).enumerate() {
                    let i = flat_idx / (dim2 * dim3 * dim4);
                    let rem1 = flat_idx % (dim2 * dim3 * dim4);
                    let j = rem1 / (dim3 * dim4);
                    let rem2 = rem1 % (dim3 * dim4);
                    let k = rem2 / dim4;
                    let l = rem2 % dim4;
                    let row_mat = x_chunk.submatrix(idx, 0, 1, in_feat).to_owned();
                    for (m, neuron) in neurons.iter().enumerate() {
                        let val = neuron.forward_mat(&row_mat);
                        out_t.data[i][j][k][l][m] = val[(0, 0)];
                    }
                }
            }
            _ => panic!("Mismatched tensor dimensions in execute_tasks for Memory"),
        }
    }

    fn create_sample_context(
        &self,
        input_sample: &DynamicTensor,
        _output_sample: &DynamicTensor,
    ) -> DynamicContext {
        match input_sample {
            DynamicTensor::Dim1(t) => DynamicContext::Ctx1D(
                crate::layers::context1d::LayerContext1D::Memory { input: t.clone() },
            ),
            DynamicTensor::Dim2(t) => DynamicContext::Ctx2D(
                crate::layers::context2d::LayerContext::Memory2D { input: t.clone() },
            ),
            DynamicTensor::Dim3(t) => DynamicContext::Ctx3D(
                crate::layers::context3d::LayerContext3D::Memory3D { input: t.clone() },
            ),
            DynamicTensor::Dim4(t) => DynamicContext::Ctx4D(
                crate::layers::context4d::LayerContext4D::Memory4D { input: t.clone() },
            ),
        }
    }

    fn output_tensor_shape(&self, input: &DynamicTensor) -> DynamicTensor {
        match input {
            DynamicTensor::Dim1(t) => DynamicTensor::Dim1(
                crate::tensor::Tensor2D::zeros(t.dim1, self.out_features),
            ),
            DynamicTensor::Dim2(t) => DynamicTensor::Dim2(
                crate::tensor::Tensor3D::zeros(t.dim1, t.dim2, self.out_features),
            ),
            DynamicTensor::Dim3(t) => DynamicTensor::Dim3(
                crate::tensor::Tensor4D::zeros(t.dim1, t.dim2, t.dim3, self.out_features),
            ),
            DynamicTensor::Dim4(t) => DynamicTensor::Dim4(
                crate::tensor::Tensor5D::zeros(t.dim1, t.dim2, t.dim3, t.dim4, self.out_features),
            ),
        }
    }
}