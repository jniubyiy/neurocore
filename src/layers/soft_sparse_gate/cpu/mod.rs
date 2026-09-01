// src/layers/soft_sparse_gate/cpu/mod.rs

use crate::compute_manager::graph::types::DynamicContext;
use crate::compute_manager::matrix_buffer::MatrixBufferHandle;
use crate::layers::buffered_context::BufferedContext;
use crate::layers::UniversalLayerBuffered;
use crate::model_plan::param_store::ParamSlice;

use super::super::soft_sparse_gate::SoftSparseGate;

impl UniversalLayerBuffered for SoftSparseGate {
    fn forward_buffered(
        &self,
        input: &MatrixBufferHandle,
        output: &MatrixBufferHandle,
        params: &MatrixBufferHandle,
        slice: &ParamSlice,
    ) {
        let rows = input.rows();
        let cols = input.cols();
        let ids = [input.id(), output.id(), params.id()];
        input.memory().write().unwrap().with_cpu_slices_mut(&ids, |slices| {
            let (first, rest) = slices.split_at_mut(1);
            let x: &[f32] = &*first[0];
            let (second, third) = rest.split_at_mut(1);
            let y: &mut [f32] = &mut *second[0];
            let p: &[f32] = &*third[0];

            let thresholds = &p[slice.start..slice.start + self.in_features];
            let tmp = self.temperature;

            for c in 0..cols {
                let threshold = thresholds[c];
                for r in 0..rows {
                    let idx = c * rows + r;
                    let x_val = x[idx];
                    let abs_x = x_val.abs();
                    let z = (abs_x - threshold) / tmp;
                    let s = 1.0 / (1.0 + (-z).exp());
                    y[idx] = x_val * s;
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
            BufferedContext::SoftSparseGate { input } => input,
            _ => panic!("Expected SoftSparseGate context"),
        };

        let rows = grad_output.rows();
        let cols = grad_output.cols();
        let ids = [
            input_handle.id(),
            grad_output.id(),
            grad_input.id(),
            params.id(),
            grad_params.id(),
        ];
        input_handle.memory().write().unwrap().with_cpu_slices_mut(&ids, |slices| {
            let (first, rest) = slices.split_at_mut(1);
            let x: &[f32] = &*first[0];
            let (second, rest) = rest.split_at_mut(1);
            let go: &[f32] = &*second[0];
            let (third, rest) = rest.split_at_mut(1);
            let gi: &mut [f32] = &mut *third[0];
            let (fourth, fifth) = rest.split_at_mut(1);
            let p: &[f32] = &*fourth[0];
            let gp: &mut [f32] = &mut *fifth[0];

            let thresholds = &p[slice.start..slice.start + self.in_features];
            let tmp = self.temperature;

            for c in 0..cols {
                let threshold = thresholds[c];
                let mut d_thr = 0.0f32;
                for r in 0..rows {
                    let idx = c * rows + r;
                    let x_val = x[idx];
                    let abs_x = x_val.abs();
                    let z = (abs_x - threshold) / tmp;
                    let s = 1.0 / (1.0 + (-z).exp());
                    let ds = s * (1.0 - s) / tmp;
                    let df_dx = s + abs_x * ds;

                    gi[idx] = go[idx] * df_dx;

                    // Градиент по порогам: d_s_dthr = -ds
                    d_thr += -go[idx] * x_val * ds;
                }
                gp[slice.start + c] = d_thr;
            }
        });
    }

    fn param_len(&self) -> usize {
        self.in_features
    }

    fn input_features(&self) -> usize {
        self.in_features
    }

    fn output_features(&self) -> usize {
        self.in_features
    }
}