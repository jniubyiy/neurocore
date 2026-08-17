// src/layers/identity/identity.rs

use crate::compute_manager::graph::types::DynamicContext;
use crate::compute_manager::matrix_buffer::MatrixBufferHandle;
use crate::layers::{UniversalLayer, UniversalLayerBuffered};
use crate::model_plan::param_store::ParamSlice;

pub struct Identity;

impl Identity {
    pub fn new() -> Self {
        Self
    }
}

impl UniversalLayer for Identity {
    fn as_identity(&self) -> Option<&Identity> {
        Some(self)
    }
}

impl UniversalLayerBuffered for Identity {
    fn forward_buffered(
        &self,
        input: &MatrixBufferHandle,
        output: &MatrixBufferHandle,
        _params: &[f32],
        _slice: &ParamSlice,
    ) {
        let src_guard = input.read();
        let src = src_guard.as_slice().expect("Identity forward: expected CPU buffer");

        let mut dst_guard = output.write();
        let dst = dst_guard.as_slice_mut().expect("Identity forward: expected CPU buffer");

        debug_assert_eq!(src.len(), dst.len());
        dst.copy_from_slice(src);
    }

    fn backward_buffered(
        &self,
        _ctx: &DynamicContext,
        grad_output: &MatrixBufferHandle,
        grad_input: &MatrixBufferHandle,
        _params: &[f32],
        _slice: &ParamSlice,
        _grad_params: &MatrixBufferHandle,
    ) {
        let go_guard = grad_output.read();
        let go = go_guard.as_slice().expect("Identity backward: expected CPU buffer");

        let mut gi_guard = grad_input.write();
        let gi = gi_guard.as_slice_mut().expect("Identity backward: expected CPU buffer");

        debug_assert_eq!(go.len(), gi.len());
        gi.copy_from_slice(go);
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