use std::any::Any;

use crate::compute_manager::matrix_buffer::MatrixBufferHandle;
use crate::plans::optimizer_plan::cube::OptimizerCube;

use super::super::scale_gradient::ScaleGradient;

impl OptimizerCube for ScaleGradient {
    fn state_size_per_param(&self) -> usize {
        0
    }

    fn apply_buffered_handle(
        &self,
        _params: &MatrixBufferHandle,
        grads: &MatrixBufferHandle,
        _state: &MatrixBufferHandle,
    ) {
        assert!(!grads.is_gpu(), "ScaleGradient: grads must be CPU");
        let mut grad_guard = grads.write();
        let grad_slice = grad_guard.as_slice_mut().expect("ScaleGradient: expected CPU buffer");
        for g in grad_slice.iter_mut() {
            *g *= self.factor;
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}