// src/layers/leaky_relu/leaky_relu.rs

use crate::compute_manager::graph::types::DynamicContext;
use crate::compute_manager::matrix_buffer::MatrixBufferHandle;
use crate::layers::buffered_context::BufferedContext;
use crate::layers::{UniversalLayer, UniversalLayerBuffered};
use crate::model_plan::param_store::ParamSlice;

pub struct LeakyReLU {
    pub alpha: f32,
}

impl LeakyReLU {
    pub fn new(alpha: f32) -> Self {
        Self { alpha }
    }
}

impl UniversalLayer for LeakyReLU {
    fn as_leaky_relu(&self) -> Option<&LeakyReLU> {
        Some(self)
    }
}

impl UniversalLayerBuffered for LeakyReLU {
    fn forward_buffered(
        &self,
        input: &MatrixBufferHandle,
        output: &MatrixBufferHandle,
        _params: &[f32],
        _slice: &ParamSlice,
    ) {
        let src_guard = input.read();
        let src = src_guard.as_slice().expect("LeakyReLU forward: expected CPU buffer");

        let mut dst_guard = output.write();
        let dst = dst_guard.as_slice_mut().expect("LeakyReLU forward: expected CPU buffer");

        debug_assert_eq!(src.len(), dst.len());

        for (o, &x) in dst.iter_mut().zip(src.iter()) {
            *o = if x > 0.0 { x } else { self.alpha * x };
        }
    }

    fn backward_buffered(
        &self,
        ctx: &DynamicContext,
        grad_output: &MatrixBufferHandle,
        grad_input: &MatrixBufferHandle,
        _params: &[f32],
        _slice: &ParamSlice,
    ) -> Vec<f32> {
        let DynamicContext::Buffered(bc) = ctx;
        let input_handle = match bc {
            BufferedContext::LeakyReLU { input } => input,
            _ => panic!("Expected LeakyReLU context"),
        };

        let input_guard = input_handle.read();
        let x_slice = input_guard.as_slice().expect("LeakyReLU backward: expected CPU buffer");

        let go_guard = grad_output.read();
        let go = go_guard.as_slice().expect("LeakyReLU backward: expected CPU buffer");

        let mut gi_guard = grad_input.write();
        let gi = gi_guard.as_slice_mut().expect("LeakyReLU backward: expected CPU buffer");

        debug_assert_eq!(go.len(), gi.len());
        debug_assert_eq!(go.len(), x_slice.len());

        for idx in 0..go.len() {
            let x_val = x_slice[idx];
            let derivative = if x_val > 0.0 { 1.0 } else { self.alpha };
            gi[idx] = go[idx] * derivative;
        }

        Vec::new()
    }

    fn param_len(&self) -> usize {
        0
    }

    fn input_features(&self) -> usize {
        0
    }

    fn output_features(&self) -> usize {
        0
    }
}