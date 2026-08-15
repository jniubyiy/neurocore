// src/layers/tanh/tanh.rs

use crate::compute_manager::graph::types::DynamicContext;
use crate::compute_manager::matrix_buffer::MatrixBufferHandle;
use crate::layers::buffered_context::BufferedContext;
use crate::layers::UniversalLayer;
use crate::layers::UniversalLayerBuffered;
use crate::model_plan::param_store::ParamSlice;

pub struct Tanh;

impl Tanh {
    pub fn new() -> Self {
        Self
    }
}

impl UniversalLayer for Tanh {
    fn as_tanh(&self) -> Option<&Tanh> {
        Some(self)
    }
}

impl UniversalLayerBuffered for Tanh {
    fn forward_buffered(
        &self,
        input: &MatrixBufferHandle,
        output: &MatrixBufferHandle,
        _params: &[f32],
        _slice: &ParamSlice,
    ) {
        let src_guard = input.read();
        let src = src_guard.as_slice().expect("Tanh forward: expected CPU buffer");

        let mut dst_guard = output.write();
        let dst = dst_guard.as_slice_mut().expect("Tanh forward: expected CPU buffer");

        debug_assert_eq!(src.len(), dst.len());

        for (o, &x) in dst.iter_mut().zip(src.iter()) {
            *o = x.tanh();
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
        // Извлекаем буферизованный контекст
        let bc = match ctx {
            DynamicContext::Buffered(bc) => bc,
            _ => panic!("Expected Buffered context"),
        };
        let output_handle = match bc {
            BufferedContext::Tanh { output } => output,
            _ => panic!("Expected Tanh context"),
        };

        let output_guard = output_handle.read();
        let y_slice = output_guard.as_slice().expect("Tanh backward: expected CPU buffer");

        let go_guard = grad_output.read();
        let go = go_guard.as_slice().expect("Tanh backward: expected CPU buffer");

        let mut gi_guard = grad_input.write();
        let gi = gi_guard.as_slice_mut().expect("Tanh backward: expected CPU buffer");

        debug_assert_eq!(go.len(), gi.len());
        debug_assert_eq!(go.len(), y_slice.len());

        for idx in 0..go.len() {
            let y_val = y_slice[idx];
            gi[idx] = go[idx] * (1.0 - y_val * y_val);
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