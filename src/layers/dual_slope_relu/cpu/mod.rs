// src/layers/dual_slope_relu/cpu/mod.rs

use crate::compute_manager::graph::types::DynamicContext;
use crate::compute_manager::matrix_buffer::MatrixBufferHandle;
use crate::layers::buffered_context::BufferedContext;
use crate::layers::UniversalLayerBuffered;
use crate::model_plan::param_store::ParamSlice;

use super::super::dual_slope_relu::DualSlopeReLU;

impl UniversalLayerBuffered for DualSlopeReLU {
    fn forward_buffered(
        &self,
        input: &MatrixBufferHandle,
        output: &MatrixBufferHandle,
        params: &MatrixBufferHandle,
        slice: &ParamSlice,
    ) {
        let rows = input.rows();
        let cols = input.cols();
        debug_assert_eq!(cols, self.features);
        debug_assert!(
            slice.start + self.param_len() <= params.rows() * params.cols(),
            "DualSlopeReLU: parameter slice out of bounds"
        );

        let ids = [input.id(), output.id(), params.id()];
        input.memory().write().unwrap().with_cpu_slices_mut(&ids, |slices| {
            let (first, rest) = slices.split_at_mut(1);
            let x: &[f32] = &*first[0];
            let (second, rest) = rest.split_at_mut(1);
            let y: &mut [f32] = &mut *second[0];
            let p: &[f32] = &*rest[0];

            let alpha_start = slice.start;
            let beta_start = alpha_start + self.features;

            for c in 0..cols {
                let alpha = p[alpha_start + c];
                let beta = p[beta_start + c];
                for r in 0..rows {
                    let idx = c * rows + r;
                    let x_val = x[idx];
                    y[idx] = if x_val >= 0.0 { beta * x_val } else { alpha * x_val };
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
            BufferedContext::DualSlopeReLU { input } => input,
            _ => panic!("Expected DualSlopeReLU context"),
        };

        let rows = grad_output.rows();
        let cols = grad_output.cols();
        debug_assert_eq!(cols, self.features);
        debug_assert_eq!(rows, input_handle.rows());
        debug_assert!(
            slice.start + self.param_len() <= params.rows() * params.cols(),
            "DualSlopeReLU backward: parameter slice out of bounds"
        );
        debug_assert!(
            slice.start + self.param_len() <= grad_params.rows() * grad_params.cols(),
            "DualSlopeReLU backward: grad parameter slice out of bounds"
        );

        let ids = [
            input_handle.id(),
            grad_output.id(),
            grad_input.id(),
            params.id(),
            grad_params.id(),
        ];
        input_handle
            .memory()
            .write()
            .unwrap()
            .with_cpu_slices_mut(&ids, |slices| {
                let (first, rest) = slices.split_at_mut(1);
                let x: &[f32] = &*first[0];
                let (second, rest) = rest.split_at_mut(1);
                let go: &[f32] = &*second[0];
                let (third, rest) = rest.split_at_mut(1);
                let gi: &mut [f32] = &mut *third[0];
                let (fourth, rest) = rest.split_at_mut(1);
                let p: &[f32] = &*fourth[0];
                let gp: &mut [f32] = &mut *rest[0];

                let alpha_start = slice.start;
                let beta_start = alpha_start + self.features;

                let mut grad_alpha = vec![0.0f32; self.features];
                let mut grad_beta = vec![0.0f32; self.features];

                for c in 0..cols {
                    let alpha = p[alpha_start + c];
                    let beta = p[beta_start + c];
                    let mut d_alpha_acc = 0.0;
                    let mut d_beta_acc = 0.0;
                    for r in 0..rows {
                        let idx = c * rows + r;
                        let x_val = x[idx];
                        let gout = go[idx];
                        if x_val >= 0.0 {
                            gi[idx] = gout * beta;
                            d_beta_acc += gout * x_val;
                        } else {
                            gi[idx] = gout * alpha;
                            d_alpha_acc += gout * x_val;
                        }
                    }
                    grad_alpha[c] = d_alpha_acc;
                    grad_beta[c] = d_beta_acc;
                }

                // записываем градиенты в общий буфер
                for c in 0..self.features {
                    gp[alpha_start + c] = grad_alpha[c];
                    gp[beta_start + c] = grad_beta[c];
                }
            });
    }

    fn param_len(&self) -> usize {
        2 * self.features
    }

    fn input_features(&self) -> usize {
        self.features
    }

    fn output_features(&self) -> usize {
        self.features
    }
}