// src/layers/learnable_softplus/cpu/mod.rs

use crate::compute_manager::graph::types::DynamicContext;
use crate::compute_manager::matrix_buffer::MatrixBufferHandle;
use crate::layers::buffered_context::BufferedContext;
use crate::layers::UniversalLayerBuffered;
use crate::model_plan::param_store::ParamSlice;

use super::super::learnable_softplus::LearnableSoftplus;

impl UniversalLayerBuffered for LearnableSoftplus {
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
            "LearnableSoftplus: parameter slice out of bounds"
        );

        let ids = [input.id(), output.id(), params.id()];
        input.memory().write().unwrap().with_cpu_slices_mut(&ids, |slices| {
            let (first, rest) = slices.split_at_mut(1);
            let x: &[f32] = &*first[0];
            let (second, rest) = rest.split_at_mut(1);
            let y: &mut [f32] = &mut *second[0];
            let p: &[f32] = &*rest[0];

            let beta_start = slice.start;
            let theta_start = beta_start + self.features;

            for c in 0..cols {
                let beta = p[beta_start + c];
                let theta = p[theta_start + c];
                for r in 0..rows {
                    let idx = c * rows + r;
                    let x_val = x[idx];
                    let shifted = beta * (x_val - theta);
                    // численно стабильно: используем log1p(exp(shifted))
                    y[idx] = (1.0 / beta) * (shifted.exp().ln_1p());
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
            BufferedContext::LearnableSoftplus { input } => input,
            _ => panic!("Expected LearnableSoftplus context"),
        };

        let rows = grad_output.rows();
        let cols = grad_output.cols();
        debug_assert_eq!(cols, self.features);
        debug_assert_eq!(rows, input_handle.rows());
        debug_assert!(
            slice.start + self.param_len() <= params.rows() * params.cols(),
            "LearnableSoftplus backward: parameter slice out of bounds"
        );
        debug_assert!(
            slice.start + self.param_len() <= grad_params.rows() * grad_params.cols(),
            "LearnableSoftplus backward: grad parameter slice out of bounds"
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

                let beta_start = slice.start;
                let theta_start = beta_start + self.features;

                let mut grad_beta = vec![0.0f32; self.features];
                let mut grad_theta = vec![0.0f32; self.features];

                for c in 0..cols {
                    let beta = p[beta_start + c];
                    let theta = p[theta_start + c];
                    let mut d_beta_acc = 0.0;
                    let mut d_theta_acc = 0.0;
                    for r in 0..rows {
                        let idx = c * rows + r;
                        let x_val = x[idx];
                        let gout = go[idx];
                        let shifted = beta * (x_val - theta);
                        let sigmoid = 1.0 / (1.0 + (-shifted).exp());
                        let y_val = (1.0 / beta) * shifted.exp().ln_1p();

                        // градиент по входу
                        gi[idx] = gout * sigmoid;

                        // градиент по beta: (d y / d beta) = (x-theta)*sigmoid - y/beta
                        let d_beta = (x_val - theta) * sigmoid - y_val / beta;
                        d_beta_acc += gout * d_beta;

                        // градиент по theta: d y / d theta = -sigmoid
                        let d_theta = -sigmoid;
                        d_theta_acc += gout * d_theta;
                    }
                    grad_beta[c] = d_beta_acc;
                    grad_theta[c] = d_theta_acc;
                }

                for c in 0..self.features {
                    gp[beta_start + c] = grad_beta[c];
                    gp[theta_start + c] = grad_theta[c];
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