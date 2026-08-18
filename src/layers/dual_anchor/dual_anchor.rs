// src/layers/dual_anchor/dual_anchor.rs

use crate::compute_manager::graph::types::DynamicContext;
use crate::compute_manager::matrix_buffer::MatrixBufferHandle;
use crate::layers::buffered_context::BufferedContext;
use crate::layers::{UniversalLayer, UniversalLayerBuffered};
use crate::model_plan::param_store::ParamSlice;

pub struct DualAnchor {
    pub features: usize,
}

impl DualAnchor {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        assert_eq!(in_features, out_features,
            "DualAnchor: in_features must equal out_features");
        Self { features: in_features }
    }
}

impl UniversalLayer for DualAnchor {
    fn as_dual_anchor(&self) -> Option<&DualAnchor> {
        Some(self)
    }

    fn param_len(&self) -> usize {
        2 * self.features + 1
    }

    fn input_features(&self) -> usize {
        self.features
    }

    fn output_features(&self) -> usize {
        self.features
    }
}

impl UniversalLayerBuffered for DualAnchor {
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
        input.memory().lock().unwrap().with_cpu_slices_mut(&ids, |slices| {
            let (first, rest) = slices.split_at_mut(1);
            let x: &[f32] = &*first[0];
            let (second, third) = rest.split_at_mut(1);
            let y: &mut [f32] = &mut *second[0];
            let p: &[f32] = &*third[0];

            let base = slice.start;
            let min_vals = &p[base..base + self.features];
            let max_vals = &p[base + self.features..base + 2 * self.features];
            let alpha = p[base + 2 * self.features];

            for c in 0..cols {
                let min_v = min_vals[c];
                let max_v = max_vals[c];
                for r in 0..rows {
                    let idx = c * rows + r;
                    let x_val = x[idx];
                    let d_min = (x_val - min_v).abs();
                    let d_max = (x_val - max_v).abs();
                    let closest = if d_min <= d_max { min_v } else { max_v };
                    y[idx] = x_val + alpha * (closest - x_val);
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
            BufferedContext::DualAnchor1D { input } => input,
            _ => panic!("Expected DualAnchor1D context"),
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
        input_handle.memory().lock().unwrap().with_cpu_slices_mut(&ids, |slices| {
            let (first, rest) = slices.split_at_mut(1);
            let x: &[f32] = &*first[0];
            let (second, rest) = rest.split_at_mut(1);
            let go: &[f32] = &*second[0];
            let (third, rest) = rest.split_at_mut(1);
            let gi: &mut [f32] = &mut *third[0];
            let (fourth, fifth) = rest.split_at_mut(1);
            let p: &[f32] = &*fourth[0];
            let gp: &mut [f32] = &mut *fifth[0];

            let base = slice.start;
            let min_vals = &p[base..base + self.features];
            let max_vals = &p[base + self.features..base + 2 * self.features];
            let alpha = p[base + 2 * self.features];

            let mut d_alpha_total = 0.0f32;

            for c in 0..cols {
                let min_v = min_vals[c];
                let max_v = max_vals[c];
                let mut d_min_acc = 0.0f32;
                let mut d_max_acc = 0.0f32;

                for r in 0..rows {
                    let idx = c * rows + r;
                    let x_val = x[idx];
                    let d_min_abs = (x_val - min_v).abs();
                    let d_max_abs = (x_val - max_v).abs();
                    let is_min = d_min_abs <= d_max_abs;
                    let gout = go[idx];

                    gi[idx] = gout * (1.0 - alpha);

                    if is_min {
                        d_min_acc += gout * alpha;
                        d_alpha_total += gout * (min_v - x_val);
                    } else {
                        d_max_acc += gout * alpha;
                        d_alpha_total += gout * (max_v - x_val);
                    }
                }

                gp[base + c] = d_min_acc;
                gp[base + self.features + c] = d_max_acc;
            }

            gp[base + 2 * self.features] = d_alpha_total;
        });
    }

    fn param_len(&self) -> usize {
        2 * self.features + 1
    }

    fn input_features(&self) -> usize {
        self.features
    }

    fn output_features(&self) -> usize {
        self.features
    }
}