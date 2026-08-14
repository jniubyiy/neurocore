// src/plans/optimizer_plan/cube.rs

use std::any::Any;
use crate::compute_manager::matrix_buffer::MatrixBufferHandle;

pub trait OptimizerCube: Send + Sync + Any {
    fn state_size_per_param(&self) -> usize;

    #[deprecated(note = "Use apply_buffered_handle for MemoryExecutor integration")]
    fn apply(&self, _params: &mut [f32], _grads: &mut [f32], _state: &mut [f32]) {
        panic!("apply is deprecated; use apply_buffered_handle");
    }

    fn apply_buffered_handle(
        &self,
        params: &MatrixBufferHandle,
        grads: &MatrixBufferHandle,
        state: &MatrixBufferHandle,
    );

    fn as_any(&self) -> &dyn Any;
}