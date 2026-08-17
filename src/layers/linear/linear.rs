// src/layers/linear/linear.rs

use crate::compute_manager::graph::types::DynamicContext;
use crate::compute_manager::matrix_buffer::MatrixBufferHandle;
use crate::layers::buffered_context::BufferedContext;
use crate::layers::{UniversalLayer, UniversalLayerBuffered};
use crate::model_plan::param_store::ParamSlice;

pub struct Linear {
    in_features: usize,
    out_features: usize,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        Self { in_features, out_features }
    }
}

impl UniversalLayer for Linear {
    fn as_linear(&self) -> Option<&Linear> {
        Some(self)
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
        grad_params: &MatrixBufferHandle,
    ) {
        let DynamicContext::Buffered(bc) = ctx;
        let input_handle = match bc {
            BufferedContext::Linear { input } => input,
            _ => panic!("Expected Linear context"),
        };

        let input_guard = input_handle.read();
        let x_slice = input_guard.as_slice().expect("Linear backward: expected CPU buffer");

        let in_rows = grad_input.rows();
        let in_cols = grad_input.cols();
        let out_cols = grad_output.cols();

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

        // Записываем градиенты весов и смещений прямо в глобальный буфер градиентов
        grad_params.with_cpu_data_mut(|grad_data| {
            // dw = grad_output^T * x
            for out_idx in 0..out_cols {
                for in_idx in 0..in_cols {
                    let mut sum = 0.0;
                    for r in 0..in_rows {
                        sum += go_slice[out_idx * in_rows + r] * x_slice[in_idx * in_rows + r];
                    }
                    grad_data[w_start + out_idx * in_cols + in_idx] = sum;
                }
            }

            // db = сумма по строкам grad_output
            for c in 0..out_cols {
                let mut sum = 0.0;
                for r in 0..in_rows {
                    sum += go_slice[c * in_rows + r];
                }
                grad_data[b_start + c] = sum;
            }
        });
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