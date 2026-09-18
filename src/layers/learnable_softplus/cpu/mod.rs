// src/layers/learnable_softplus/cpu/mod.rs

use crate::compute_manager::core::dynamic_context::DynamicContext;
use crate::compute_manager::operators_v2::memory_v2::buffer::{MatrixBufferHandle, TempMatrixPool};
use crate::layers::buffered_context::BufferedContext;
use crate::layers::UniversalLayerBuffered;
use crate::model_plan::param_store::ParamSlice;

use super::super::learnable_softplus::LearnableSoftplus;

// ============================================================================
// Параметризация β и математическое обоснование
// ============================================================================
//
//   β = exp(log_beta)
//
// Это ЕДИНСТВЕННАЯ гладкая параметризация, устраняющая сингулярность 1/β
// в ∂y/∂β:
//
//   y = softplus(β·u)/β,   u = x − θ
//   ∂y/∂β = (σ(β·u)·u − y)/β
//   ∂y/∂log_beta = ∂y/∂β · β = σ(β·u)·u − y    ← без 1/β
// ============================================================================

const BETA_MIN: f32 = 1e-3;
const LOG_BETA_MIN: f32 = -6.907755;  // ln(1e-3)

#[inline]
fn softplus_stable(z: f32) -> f32 {
    if z > 0.0 {
        z + (-z).exp().ln_1p()
    } else {
        z.exp().ln_1p()
    }
}

#[inline]
fn effective_beta(log_beta: f32) -> (f32, f32, bool) {
    if log_beta < LOG_BETA_MIN {
        (LOG_BETA_MIN, BETA_MIN, true)
    } else {
        (log_beta, log_beta.exp(), false)
    }
}

impl UniversalLayerBuffered for LearnableSoftplus {
    fn forward_buffered(
        &self,
        input: &MatrixBufferHandle,
        output: &MatrixBufferHandle,
        params: &MatrixBufferHandle,
        slice: &ParamSlice,
        _pool: &mut TempMatrixPool,
    ) -> BufferedContext {
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

            let log_beta_start = slice.start;
            let theta_start = log_beta_start + self.features;

            for c in 0..cols {
                let log_beta = p[log_beta_start + c];
                let (_, beta, _) = effective_beta(log_beta);
                let inv_beta = 1.0 / beta;
                let theta = p[theta_start + c];

                for r in 0..rows {
                    let idx = c * rows + r;
                    let x_val = x[idx];
                    let shifted = beta * (x_val - theta);
                    y[idx] = inv_beta * softplus_stable(shifted);
                }
            }
        });

        BufferedContext::LearnableSoftplus {
            input: input.clone(),
        }
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

                let log_beta_start = slice.start;
                let theta_start = log_beta_start + self.features;

                let mut grad_log_beta = vec![0.0f32; self.features];
                let mut grad_theta = vec![0.0f32; self.features];

                for c in 0..cols {
                    let log_beta = p[log_beta_start + c];
                    let (_, beta, is_clamped) = effective_beta(log_beta);
                    let inv_beta = 1.0 / beta;
                    let theta = p[theta_start + c];

                    let mut d_log_beta_acc = 0.0f32;
                    let mut d_theta_acc = 0.0f32;

                    for r in 0..rows {
                        let idx = c * rows + r;
                        let x_val = x[idx];
                        let gout = go[idx];
                        let u = x_val - theta;

                        let shifted = beta * u;
                        let sigmoid = 1.0 / (1.0 + (-shifted).exp());
                        let y_val = inv_beta * softplus_stable(shifted);

                        // ∂y/∂x = σ(β·u)
                        gi[idx] = gout * sigmoid;

                        // ∂y/∂log_beta = σ(β·u)·u − y
                        if !is_clamped {
                            let d_log_beta = sigmoid * u - y_val;
                            d_log_beta_acc += gout * d_log_beta;
                        }

                        // ∂y/∂θ = −σ(β·u)
                        let d_theta = -sigmoid;
                        d_theta_acc += gout * d_theta;
                    }

                    grad_log_beta[c] = d_log_beta_acc;
                    grad_theta[c] = d_theta_acc;
                }

                for c in 0..self.features {
                    gp[log_beta_start + c] = grad_log_beta[c];
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