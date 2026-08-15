// src/layers/relu/relu.rs

use crate::compute_manager::graph::types::DynamicContext;
use crate::compute_manager::matrix_buffer::MatrixBufferHandle;
use crate::layers::buffered_context::BufferedContext;
use crate::layers::{UniversalLayer, UniversalLayerBuffered};
use crate::model_plan::param_store::ParamSlice;

pub struct ReLU;

impl ReLU {
    pub fn new() -> Self {
        Self
    }
}

impl UniversalLayer for ReLU {
    fn as_relu(&self) -> Option<&ReLU> {
        Some(self)
    }
}

impl UniversalLayerBuffered for ReLU {
    fn forward_buffered(
        &self,
        input: &MatrixBufferHandle,
        output: &MatrixBufferHandle,
        _params: &[f32],
        _slice: &ParamSlice,
    ) {
        let src_guard = input.read();
        let src = src_guard.as_slice().expect("ReLU forward: expected CPU buffer");

        let mut dst_guard = output.write();
        let dst = dst_guard.as_slice_mut().expect("ReLU forward: expected CPU buffer");

        debug_assert_eq!(src.len(), dst.len());

        for (o, &x) in dst.iter_mut().zip(src.iter()) {
            *o = x.max(0.0);
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
            BufferedContext::ReLU { input } => input,
            _ => panic!("Expected ReLU context"),
        };

        let input_guard = input_handle.read();
        let x_slice = input_guard.as_slice().expect("ReLU backward: expected CPU buffer");

        let go_guard = grad_output.read();
        let go = go_guard.as_slice().expect("ReLU backward: expected CPU buffer");

        let mut gi_guard = grad_input.write();
        let gi = gi_guard.as_slice_mut().expect("ReLU backward: expected CPU buffer");

        debug_assert_eq!(go.len(), gi.len());
        debug_assert_eq!(go.len(), x_slice.len());

        for idx in 0..go.len() {
            gi[idx] = if x_slice[idx] > 0.0 { go[idx] } else { 0.0 };
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