// src/layers/linear/linear.rs

use crate::compute_manager::graph::types::DynamicContext;
use crate::compute_manager::matrix_buffer::MatrixBufferHandle;
use crate::layers::buffered_context::BufferedContext;
use crate::layers::UniversalLayer;
use crate::layers::UniversalLayerBuffered;
use crate::model_plan::param_store::ParamSlice;
use crate::layers::mat_context::MatContext;
use faer::Mat;

pub struct Linear {
    in_features: usize,
    out_features: usize,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        Self { in_features, out_features }
    }

    pub(crate) fn get_weight_matrix_and_bias(
        &self,
        params: &[f32],
        slice: &ParamSlice,
    ) -> (Mat<f32>, Vec<f32>) {
        let in_feat = self.in_features;
        let out_feat = self.out_features;
        let w_start = slice.start;
        let b_start = w_start + in_feat * out_feat;

        let weight = Mat::from_fn(out_feat, in_feat, |r, c| {
            params[w_start + r * in_feat + c]
        });
        let bias = params[b_start..b_start + out_feat].to_vec();
        (weight, bias)
    }
}

// ---------------------------------------------------------------------------
// Старая реализация UniversalLayer (оставлена для GPU и обратной совместимости)
// ---------------------------------------------------------------------------

impl UniversalLayer for Linear {
    fn forward_mat(
        &self,
        input: &Mat<f32>,
        params: &[f32],
        slice: &ParamSlice,
    ) -> (Mat<f32>, DynamicContext) {
        let (weight, bias) = self.get_weight_matrix_and_bias(params, slice);
        let batch = input.nrows();
        let mut output = input * weight.transpose();
        output += Mat::from_fn(batch, self.out_features, |_, j| bias[j]);

        let ctx = DynamicContext::Mat(MatContext::Linear { input: input.clone() });
        (output, ctx)
    }

    fn backward_mat(
        &self,
        ctx: &DynamicContext,
        delta: &Mat<f32>,
        params: &[f32],
        slice: &ParamSlice,
    ) -> (Mat<f32>, Vec<f32>) {
        let x_mat = match ctx {
            DynamicContext::Mat(MatContext::Linear { input }) => input.clone(),
            _ => panic!("Expected Linear context"),
        };
        let (weight, _) = self.get_weight_matrix_and_bias(params, slice);
        let dx = delta * &weight;
        let dw = delta.transpose() * &x_mat;
        let batch = delta.nrows();
        let mut db = vec![0.0f32; self.out_features];
        for r in 0..batch {
            for c in 0..self.out_features {
                db[c] += delta[(r, c)];
            }
        }

        let mut grad = Vec::with_capacity(<Self as UniversalLayer>::param_len(self));
        for r in 0..self.out_features {
            for c in 0..self.in_features {
                grad.push(dw[(r, c)]);
            }
        }
        grad.extend_from_slice(&db);
        (dx, grad)
    }

    fn param_len(&self) -> usize {
        self.in_features * self.out_features + self.out_features
    }

    fn input_features(&self) -> usize { self.in_features }
    fn output_features(&self) -> usize { self.out_features }

    fn total_tasks(&self, batch_size: usize) -> usize { batch_size }

    fn execute_tasks(
        &self,
        input: &Mat<f32>,
        output: &mut Mat<f32>,
        task_offset: usize,
        task_count: usize,
        params: &[f32],
        slice: &ParamSlice,
    ) {
        let (weight, bias) = self.get_weight_matrix_and_bias(params, slice);
        let chunk = input.submatrix(task_offset, 0, task_count, self.in_features);
        let mut out_chunk = &chunk * weight.transpose();
        out_chunk += Mat::from_fn(task_count, self.out_features, |_, j| bias[j]);
        for r in 0..task_count {
            for c in 0..self.out_features {
                output[(task_offset + r, c)] = out_chunk[(r, c)];
            }
        }
    }

    fn create_sample_context(
        &self,
        input_sample: &Mat<f32>,
        _output_sample: &Mat<f32>,
    ) -> DynamicContext {
        DynamicContext::Mat(MatContext::Linear { input: input_sample.clone() })
    }

    fn output_mat_shape(&self, batch_size: usize) -> Mat<f32> {
        Mat::zeros(batch_size, self.out_features)
    }

    fn as_linear(&self) -> Option<&Linear> {
        Some(self)
    }
}

// ---------------------------------------------------------------------------
// Новая реализация UniversalLayerBuffered (CPU‑путь с управляемыми буферами)
// ---------------------------------------------------------------------------

impl UniversalLayerBuffered for Linear {
    fn forward_buffered(
        &self,
        input: &MatrixBufferHandle,
        output: &MatrixBufferHandle,
        params: &[f32],
        slice: &ParamSlice,
    ) {
        let in_rows = input.rows();
        let in_cols = input.cols();
        let out_cols = self.out_features;

        let input_guard = input.read();
        let input_slice = input_guard.as_slice().expect("Linear forward: expected CPU buffer");

        let mut output_guard = output.write();
        let output_slice = output_guard.as_slice_mut().expect("Linear forward: expected CPU buffer");

        debug_assert_eq!(input_slice.len(), in_rows * in_cols);
        debug_assert_eq!(output_slice.len(), in_rows * out_cols);

        let w_start = slice.start;
        let b_start = w_start + in_cols * out_cols;

        // output[r, c] = bias[c] + sum_k input[r, k] * weight[c, k]
        for r in 0..in_rows {
            for c in 0..out_cols {
                let mut sum = params[b_start + c];
                for k in 0..in_cols {
                    sum += input_slice[k * in_rows + r] * params[w_start + c * in_cols + k];
                }
                output_slice[c * in_rows + r] = sum;
            }
        }
    }

    fn backward_buffered(
        &self,
        ctx: &DynamicContext,
        grad_output: &MatrixBufferHandle,
        grad_input: &MatrixBufferHandle,
        params: &[f32],
        slice: &ParamSlice,
    ) -> Vec<f32> {
        // Извлекаем буферизованный контекст
        let bc = match ctx {
            DynamicContext::Buffered(bc) => bc,
            _ => panic!("Expected Buffered context"),
        };
        let input_handle = match bc {
            BufferedContext::Linear { input } => input,
            _ => panic!("Expected Linear context"),
        };

        let input_guard = input_handle.read();
        let x_slice = input_guard.as_slice().expect("Linear backward: expected CPU buffer");

        let in_rows = grad_input.rows();
        let in_cols = grad_input.cols();      // == self.in_features
        let out_cols = grad_output.cols();    // == self.out_features

        let go_guard = grad_output.read();
        let go_slice = go_guard.as_slice().expect("Linear backward: expected CPU buffer");

        let mut gi_guard = grad_input.write();
        let gi_slice = gi_guard.as_slice_mut().expect("Linear backward: expected CPU buffer");

        debug_assert_eq!(go_slice.len(), in_rows * out_cols);
        debug_assert_eq!(gi_slice.len(), in_rows * in_cols);

        let w_start = slice.start;
        let b_start = w_start + in_cols * out_cols;

        // dx = grad_output * weight
        for r in 0..in_rows {
            for c in 0..in_cols {
                let mut sum = 0.0;
                for k in 0..out_cols {
                    sum += go_slice[k * in_rows + r] * params[w_start + k * in_cols + c];
                }
                gi_slice[c * in_rows + r] = sum;
            }
        }

        // dw = grad_output^T * x
        let mut dw = vec![0.0f32; in_cols * out_cols];
        for out_idx in 0..out_cols {
            for in_idx in 0..in_cols {
                let mut sum = 0.0;
                for r in 0..in_rows {
                    sum += go_slice[out_idx * in_rows + r] * x_slice[in_idx * in_rows + r];
                }
                dw[out_idx * in_cols + in_idx] = sum;
            }
        }

        // db = сумма по строкам grad_output
        let mut db = vec![0.0f32; out_cols];
        for c in 0..out_cols {
            let mut sum = 0.0;
            for r in 0..in_rows {
                sum += go_slice[c * in_rows + r];
            }
            db[c] = sum;
        }

        // Собираем градиенты параметров: сначала dw, затем db
        let mut grad = Vec::with_capacity(self.in_features * self.out_features + self.out_features);
        grad.extend_from_slice(&dw);
        grad.extend_from_slice(&db);
        grad
    }

    fn param_len(&self) -> usize {
        self.in_features * self.out_features + self.out_features
    }

    fn input_features(&self) -> usize {
        self.in_features
    }

    fn output_features(&self) -> usize {
        self.out_features
    }
}