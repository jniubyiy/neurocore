// src/layers/sigmoid/cpu/mod.rs

use crate::compute_manager::graph::types::DynamicContext;
use crate::compute_manager::matrix_buffer::MatrixBufferHandle;
use crate::layers::buffered_context::BufferedContext;
use crate::layers::UniversalLayerBuffered;
use crate::model_plan::param_store::ParamSlice;

use super::super::sigmoid::Sigmoid;

impl UniversalLayerBuffered for Sigmoid {
    fn forward_buffered(
        &self,
        input: &MatrixBufferHandle,
        output: &MatrixBufferHandle,
        _params: &MatrixBufferHandle,
        _slice: &ParamSlice,
    ) {
        let ids = [input.id(), output.id()];
        input.memory().write().unwrap().with_cpu_slices_mut(&ids, |slices| {
            let (first, rest) = slices.split_at_mut(1);
            let x: &[f32] = &*first[0];
            let y: &mut [f32] = &mut *rest[0];
            for i in 0..x.len() {
                y[i] = 1.0 / (1.0 + (-x[i]).exp());
            }
        });
    }

    fn backward_buffered(
        &self,
        ctx: &DynamicContext,
        grad_output: &MatrixBufferHandle,
        grad_input: &MatrixBufferHandle,
        _params: &MatrixBufferHandle,
        _slice: &ParamSlice,
        _grad_params: &MatrixBufferHandle,
    ) {
        let DynamicContext::Buffered(bc) = ctx;
        let output_handle = match bc {
            BufferedContext::Sigmoid { output } => output,
            _ => panic!("Expected Sigmoid context"),
        };

        let ids = [output_handle.id(), grad_output.id(), grad_input.id()];
        output_handle.memory().write().unwrap().with_cpu_slices_mut(&ids, |slices| {
            let (first, rest) = slices.split_at_mut(1);
            let y: &[f32] = &*first[0];
            let (second, rest) = rest.split_at_mut(1);
            let go: &[f32] = &*second[0];
            let gi: &mut [f32] = &mut *rest[0];
            for i in 0..go.len() {
                let y_val = y[i];
                gi[i] = go[i] * y_val * (1.0 - y_val);
            }
        });
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