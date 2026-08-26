use std::any::Any;

use crate::compute_manager::matrix_buffer::MatrixBufferHandle;
use crate::plans::optimizer_plan::cube::OptimizerCube;

use super::super::apply_update::ApplyUpdate;

impl OptimizerCube for ApplyUpdate {
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
            "ApplyUpdate: params and grads must be CPU"
        );
        let mut param_guard = params.write();
        let p_slice = param_guard.as_slice_mut().expect("ApplyUpdate: expected CPU buffer");
        let grad_guard = grads.read();
        let g_slice = grad_guard.as_slice().expect("ApplyUpdate: expected CPU buffer");

        debug_assert_eq!(p_slice.len(), g_slice.len());

        for i in 0..p_slice.len() {
            p_slice[i] -= g_slice[i];
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}