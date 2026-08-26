use std::any::Any;

use crate::compute_manager::matrix_buffer::MatrixBufferHandle;
use crate::plans::optimizer_plan::cube::OptimizerCube;

use super::super::gradient_clip::GradientClip;

impl OptimizerCube for GradientClip {
    fn state_size_per_param(&self) -> usize {
        0
    }

    fn apply_buffered_handle(
        &self,
        _params: &MatrixBufferHandle,
        grads: &MatrixBufferHandle,
        _state: &MatrixBufferHandle,
    ) {
        assert!(!grads.is_gpu(), "GradientClip: grads must be CPU");
        let mut grad_guard = grads.write();
        let g_slice = grad_guard.as_slice_mut().expect("GradientClip: expected CPU buffer");

        for g in g_slice.iter_mut() {
            if let Some(min_val) = self.min {
                *g = g.max(min_val);
            }
            if let Some(max_val) = self.max {
                *g = g.min(max_val);
            }
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}