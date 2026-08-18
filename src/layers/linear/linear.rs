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
        params: &MatrixBufferHandle,
        slice: &ParamSlice,
    ) {
        let in_rows = input.rows();
        let in_cols = input.cols();
        let out_cols = self.out_features;

        let ids = [input.id(), output.id(), params.id()];
        input.memory().lock().unwrap().with_cpu_slices_mut(&ids, |slices| {
            let (first, rest) = slices.split_at_mut(1);
            let x: &[f32] = &*first[0];
            let (second, rest) = rest.split_at_mut(1);
            let y: &mut [f32] = &mut *second[0];
            let p: &[f32] = &*rest[0];

            let w_start = slice.start;
            let b_start = w_start + in_cols * out_cols;

            // output[r, c] = bias[c] + sum_k input[r, k] * weight[c, k]
            for r in 0..in_rows {
                for c in 0..out_cols {
                    let mut sum = p[b_start + c];
                    for k in 0..in_cols {
                        sum += x[k * in_rows + r] * p[w_start + c * in_cols + k];
                    }
                    y[c * in_rows + r] = sum;
                }
            }
        });
    }

    fn backward_buffered(
        &self,
        ctx: &DynamicContext,
        grad_output: &MatrixBufferHandle,
        grad_input: &MatrixBufferHandle,
        params: &MatrixBufferHandle,
        slice: &ParamSlice,
        grad_params: &MatrixBufferHandle,
    ) {
        let DynamicContext::Buffered(bc) = ctx;
        let input_handle = match bc {
            BufferedContext::Linear { input } => input,
            _ => panic!("Expected Linear context"),
        };

        let in_rows = grad_input.rows();
        let in_cols = grad_input.cols();
        let out_cols = grad_output.cols();

        let ids = [
            input_handle.id(),
            grad_output.id(),
            grad_input.id(),
            params.id(),
            grad_params.id(),
        ];
        input_handle.memory().lock().unwrap().with_cpu_slices_mut(&ids, |slices| {
            let (first, rest) = slices.split_at_mut(1);
            let x: &[f32] = &*first[0];
            let (second, rest) = rest.split_at_mut(1);
            let go: &[f32] = &*second[0];
            let (third, rest) = rest.split_at_mut(1);
            let gi: &mut [f32] = &mut *third[0];
            let (fourth, rest) = rest.split_at_mut(1);
            let p: &[f32] = &*fourth[0];
            let gp: &mut [f32] = &mut *rest[0];

            let w_start = slice.start;
            let b_start = w_start + in_cols * out_cols;

            // dx = grad_output * weight
            for r in 0..in_rows {
                for c in 0..in_cols {
                    let mut sum = 0.0;
                    for k in 0..out_cols {
                        sum += go[k * in_rows + r] * p[w_start + k * in_cols + c];
                    }
                    gi[c * in_rows + r] = sum;
                }
            }

            // dw = grad_output^T * x
            for out_idx in 0..out_cols {
                for in_idx in 0..in_cols {
                    let mut sum = 0.0;
                    for r in 0..in_rows {
                        sum += go[out_idx * in_rows + r] * x[in_idx * in_rows + r];
                    }
                    gp[w_start + out_idx * in_cols + in_idx] = sum;
                }
            }

            // db = сумма по строкам grad_output
            for c in 0..out_cols {
                let mut sum = 0.0;
                for r in 0..in_rows {
                    sum += go[c * in_rows + r];
                }
                gp[b_start + c] = sum;
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