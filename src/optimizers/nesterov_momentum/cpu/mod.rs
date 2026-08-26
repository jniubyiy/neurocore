use std::any::Any;

use crate::compute_manager::matrix_buffer::MatrixBufferHandle;
use crate::plans::optimizer_plan::cube::OptimizerCube;

use super::super::nesterov_momentum::NesterovMomentum;

impl OptimizerCube for NesterovMomentum {
    fn state_size_per_param(&self) -> usize {
        1
    }

    fn apply_buffered_handle(
        &self,
        _params: &MatrixBufferHandle,
        grads: &MatrixBufferHandle,
        state: &MatrixBufferHandle,
    ) {
        assert!(
            !grads.is_gpu() && !state.is_gpu(),
            "NesterovMomentum: grads and state must be CPU"
        );
        let mut grad_guard = grads.write();
        let g_slice = grad_guard.as_slice_mut().expect("NesterovMomentum: expected CPU buffer");
        let mut state_guard = state.write();
        let s_slice = state_guard.as_slice_mut().expect("NesterovMomentum: expected CPU buffer");

        debug_assert_eq!(g_slice.len(), s_slice.len());

        for i in 0..g_slice.len() {
            let v_old = s_slice[i];
            let v_new = self.beta * v_old + g_slice[i];
            g_slice[i] += self.beta * v_new;
            s_slice[i] = v_new;
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}