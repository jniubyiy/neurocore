// src/layers/tanh/cpu/mod.rs

use crate::compute_manager::core::dynamic_context::DynamicContext;
use crate::compute_manager::operators_v2::memory_v2::buffer::{MatrixBufferHandle, TempMatrixPool};
use crate::layers::buffered_context::BufferedContext;
use crate::layers::UniversalLayerBuffered;
use crate::model_plan::param_store::ParamSlice;

use super::super::tanh::Tanh;

impl UniversalLayerBuffered for Tanh {
    fn forward_buffered(
        &self,
        input: &MatrixBufferHandle,
        output: &MatrixBufferHandle,
        _params: &MatrixBufferHandle,
        _slice: &ParamSlice,
        _pool: &mut TempMatrixPool,
    ) -> BufferedContext {
        let ids = [input.id(), output.id()];
        input.memory().write().unwrap().with_cpu_slices_mut(&ids, |slices| {
            let (first, rest) = slices.split_at_mut(1);
            let x: &[f32] = &*first[0];
            let y: &mut [f32] = &mut *rest[0];
            for i in 0..x.len() {
                y[i] = x[i].tanh();
            }
        });

        // Backward использует выход tanh, поэтому кладём в контекст
        // именно output.
        BufferedContext::Tanh {
            output: output.clone(),
        }
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
            BufferedContext::Tanh { output } => output,
            _ => panic!("Expected Tanh context"),
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
                gi[i] = go[i] * (1.0 - y_val * y_val);
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