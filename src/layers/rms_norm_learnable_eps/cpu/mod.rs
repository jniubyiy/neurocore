// src/layers/rms_norm_learnable_eps/cpu/mod.rs

use crate::compute_manager::graph::types::DynamicContext;
use crate::compute_manager::matrix_buffer::MatrixBufferHandle;
use crate::layers::buffered_context::BufferedContext;
use crate::layers::UniversalLayerBuffered;
use crate::model_plan::param_store::ParamSlice;

use super::super::rms_norm_learnable_eps::RMSNormWithLearnableEpsilon;

impl UniversalLayerBuffered for RMSNormWithLearnableEpsilon {
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
            "RMSNormWithLearnableEpsilon: parameter slice out of bounds"
        );

        let ids = [input.id(), output.id(), params.id()];
        input.memory().write().unwrap().with_cpu_slices_mut(&ids, |slices| {
            let (first, rest) = slices.split_at_mut(1);
            let x: &[f32] = &*first[0];
            let (second, rest) = rest.split_at_mut(1);
            let y: &mut [f32] = &mut *second[0];
            let p: &[f32] = &*rest[0];

            let gamma_start = slice.start;
            let eps_start = gamma_start + self.features;

            // Вычисляем mean(x^2) для каждой строки
            let mut mean_sq = vec![0.0f32; rows];
            for r in 0..rows {
                let mut sum_sq = 0.0f32;
                for c in 0..cols {
                    let idx = c * rows + r;
                    let v = x[idx];
                    sum_sq += v * v;
                }
                mean_sq[r] = sum_sq / cols as f32;
            }

            for c in 0..cols {
                let gamma = p[gamma_start + c];
                let eps = p[eps_start + c];
                for r in 0..rows {
                    let idx = c * rows + r;
                    let denom = (mean_sq[r] + eps).sqrt();
                    y[idx] = (x[idx] / denom) * gamma;
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
            BufferedContext::RMSNormWithLearnableEpsilon { input } => input,
            _ => panic!("Expected RMSNormWithLearnableEpsilon context"),
        };

        let rows = grad_output.rows();
        let cols = grad_output.cols();
        debug_assert_eq!(cols, self.features);
        debug_assert_eq!(rows, input_handle.rows());

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

                let gamma_start = slice.start;
                let eps_start = gamma_start + self.features;

                // Вычисляем mean_sq по строкам
                let mut mean_sq = vec![0.0f32; rows];
                for r in 0..rows {
                    let mut sum_sq = 0.0f32;
                    for c in 0..cols {
                        let idx = c * rows + r;
                        let v = x[idx];
                        sum_sq += v * v;
                    }
                    mean_sq[r] = sum_sq / cols as f32;
                }

                let mut grad_gamma = vec![0.0f32; self.features];
                let mut grad_eps = vec![0.0f32; self.features];

                for c in 0..cols {
                    let gamma = p[gamma_start + c];
                    let eps = p[eps_start + c];

                    let mut d_gamma_acc = 0.0f32;
                    let mut d_eps_acc = 0.0f32;

                    for r in 0..rows {
                        let idx = c * rows + r;
                        let x_val = x[idx];
                        let gout = go[idx];
                        let denom = (mean_sq[r] + eps).sqrt();
                        let denom3 = denom * denom * denom;

                        // Градиент по входу
                        let term1 = gamma / denom;
                        let sum_j = {
                            // sum over all features j of gout_j * x_j / denom_j
                            let mut s = 0.0f32;
                            for j in 0..cols {
                                let idx_j = j * rows + r;
                                let gamma_j = p[gamma_start + j];
                                let eps_j = p[eps_start + j];
                                let denom_j = (mean_sq[r] + eps_j).sqrt();
                                s += go[idx_j] * gamma_j * x[idx_j] / denom_j;
                            }
                            s
                        };
                        let term2 = (1.0 / cols as f32) * x_val / (denom3) * sum_j;
                        gi[idx] = gout * term1 - term2;

                        // Градиенты по параметрам
                        d_gamma_acc += gout * x_val / denom;
                        d_eps_acc += -0.5 * gout * gamma * x_val / denom3;
                    }

                    grad_gamma[c] = d_gamma_acc;
                    grad_eps[c] = d_eps_acc;
                }

                for c in 0..self.features {
                    gp[gamma_start + c] = grad_gamma[c];
                    gp[eps_start + c] = grad_eps[c];
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