use std::any::Any;

use crate::compute_manager::matrix_buffer::MatrixBufferHandle;
use crate::plans::optimizer_plan::cube::OptimizerCube;

use super::super::add_weight_decay::AddWeightDecay;

impl OptimizerCube for AddWeightDecay {
    fn state_size_per_param(&self) -> usize {
        0
    }

    fn apply_buffered_handle(
        &self,
        params: &MatrixBufferHandle,
        grads: &MatrixBufferHandle,
        _state: &MatrixBufferHandle,
    ) {
        assert!(
            !params.is_gpu() && !grads.is_gpu(),
            "AddWeightDecay: params and grads must be CPU"
        );
        let param_guard = params.read();
        let p_slice = param_guard.as_slice().expect("AddWeightDecay: expected CPU buffer");
        let mut grad_guard = grads.write();
        let g_slice = grad_guard.as_slice_mut().expect("AddWeightDecay: expected CPU buffer");

        debug_assert_eq!(p_slice.len(), g_slice.len());

        for i in 0..g_slice.len() {
            g_slice[i] += self.decay * p_slice[i];
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}