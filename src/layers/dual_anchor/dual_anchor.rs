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
        params: &[f32],
        slice: &ParamSlice,
    ) {
        let input_guard = input.read();
        let src = input_guard.as_slice().expect("DualAnchor forward: expected CPU buffer");

        let mut output_guard = output.write();
        let dst = output_guard.as_slice_mut().expect("DualAnchor forward: expected CPU buffer");

        let rows = input.rows();
        let cols = input.cols();
        debug_assert_eq!(cols, self.features);

        let base = slice.start;
        let min_vals = &params[base..base + self.features];
        let max_vals = &params[base + self.features..base + 2 * self.features];
        let alpha = params[base + 2 * self.features];

        debug_assert_eq!(src.len(), dst.len());

        for c in 0..cols {
            let min_v = min_vals[c];
            let max_v = max_vals[c];
            for r in 0..rows {
                let idx = c * rows + r;
                let x = src[idx];
                let d_min = (x - min_v).abs();
                let d_max = (x - max_v).abs();
                let closest = if d_min <= d_max { min_v } else { max_v };
                dst[idx] = x + alpha * (closest - x);
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
        let DynamicContext::Buffered(bc) = ctx;
        let input_handle = match bc {
            BufferedContext::DualAnchor1D { input } => input,
            _ => panic!("Expected DualAnchor1D context"),
        };

        let input_guard = input_handle.read();
        let x_slice = input_guard.as_slice().expect("DualAnchor backward: expected CPU buffer");

        let rows = grad_output.rows();
        let cols = grad_output.cols();
        debug_assert_eq!(cols, self.features);

        let base = slice.start;
        let min_vals = &params[base..base + self.features];
        let max_vals = &params[base + self.features..base + 2 * self.features];
        let alpha = params[base + 2 * self.features];

        let go_guard = grad_output.read();
        let go = go_guard.as_slice().expect("DualAnchor backward: expected CPU buffer");

        let mut gi_guard = grad_input.write();
        let gi = gi_guard.as_slice_mut().expect("DualAnchor backward: expected CPU buffer");

        debug_assert_eq!(go.len(), gi.len());
        debug_assert_eq!(go.len(), x_slice.len());

        let mut d_min_accum = vec![0.0f32; self.features];
        let mut d_max_accum = vec![0.0f32; self.features];
        let mut d_alpha = 0.0f32;

        for c in 0..cols {
            let min_v = min_vals[c];
            let max_v = max_vals[c];
            for r in 0..rows {
                let idx = c * rows + r;

                let x_val = x_slice[idx];
                let d_min_abs = (x_val - min_v).abs();
                let d_max_abs = (x_val - max_v).abs();
                let is_min = d_min_abs <= d_max_abs;
                let gout = go[idx];

                gi[idx] = gout * (1.0 - alpha);

                if is_min {
                    d_min_accum[c] += gout * alpha;
                    d_alpha += gout * (min_v - x_val);
                } else {
                    d_max_accum[c] += gout * alpha;
                    d_alpha += gout * (max_v - x_val);
                }
            }
        }

        let mut grad = Vec::with_capacity(2 * self.features + 1);
        grad.extend_from_slice(&d_min_accum);
        grad.extend_from_slice(&d_max_accum);
        grad.push(d_alpha);
        grad
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