// src/layers/sparse_feature_selection_gate/cpu/mod.rs

use crate::compute_manager::graph::types::DynamicContext;
use crate::compute_manager::matrix_buffer::MatrixBufferHandle;
use crate::layers::buffered_context::BufferedContext;
use crate::layers::UniversalLayerBuffered;
use crate::model_plan::param_store::ParamSlice;

use super::super::sparse_feature_selection_gate::SparseFeatureSelectionGate;

impl UniversalLayerBuffered for SparseFeatureSelectionGate {
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
        debug_assert!(slice.start + self.param_len() <= params.rows() * params.cols());

        let ids = [input.id(), output.id(), params.id()];
        input.memory().write().unwrap().with_cpu_slices_mut(&ids, |slices| {
            let (first, rest) = slices.split_at_mut(1);
            let x: &[f32] = &*first[0];
            let (second, rest) = rest.split_at_mut(1);
            let y: &mut [f32] = &mut *second[0];
            let p: &[f32] = &*rest[0];

            let base = slice.start;
            let logits_start = base;
            let temp_idx = base + self.features;
            let temperature = p[temp_idx].abs() + 1e-6; // гарантируем положительность

            for c in 0..cols {
                let mask = 1.0 / (1.0 + (-p[logits_start + c] / temperature).exp());
                for r in 0..rows {
                    let idx = c * rows + r;
                    y[idx] = x[idx] * mask;
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
            BufferedContext::SparseFeatureSelectionGate { input } => input,
            _ => panic!("Expected SparseFeatureSelectionGate context"),
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

                let base = slice.start;
                let logits_start = base;
                let temp_idx = base + self.features;
                let temperature = p[temp_idx].abs() + 1e-6;

                let mut grad_logits = vec![0.0f32; self.features];
                let mut grad_temp = 0.0f32;

                for c in 0..cols {
                    let logit = p[logits_start + c];
                    let sig = 1.0 / (1.0 + (-logit / temperature).exp());
                    let dsig_dlogit = sig * (1.0 - sig) / temperature;
                    let dsig_dtemp = -sig * (1.0 - sig) * logit / (temperature * temperature);

                    let mut d_logit_acc = 0.0;
                    let mut d_temp_acc = 0.0;
                    for r in 0..rows {
                        let idx = c * rows + r;
                        let gout = go[idx];
                        let x_val = x[idx];

                        gi[idx] = gout * sig;

                        d_logit_acc += gout * x_val * dsig_dlogit;
                        d_temp_acc += gout * x_val * dsig_dtemp;
                    }
                    grad_logits[c] = d_logit_acc;
                    grad_temp += d_temp_acc;
                }

                // записываем градиенты параметров
                for c in 0..self.features {
                    gp[logits_start + c] = grad_logits[c];
                }
                gp[temp_idx] = grad_temp;
            });
    }

    fn param_len(&self) -> usize {
        self.features + 1
    }

    fn input_features(&self) -> usize {
        self.features
    }

    fn output_features(&self) -> usize {
        self.features
    }
}