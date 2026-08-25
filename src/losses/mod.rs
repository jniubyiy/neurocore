// src/losses/mod.rs

use std::any::Any;
use std::fmt::Debug;
use crate::compute_manager::matrix_buffer::MatrixBufferHandle;

pub mod sub;

/// Элементарный кубик функции потерь (матричная версия).
pub trait ElemCube: Any + Send + Sync + Debug {
    fn in_features(&self) -> usize;
    fn out_features(&self) -> usize;
    fn as_any(&self) -> &dyn Any;
}

/// Буферизованный элементарный кубик функции потерь.
/// Работает с управляемыми буферами `MatrixBufferHandle` (CPU).
pub trait BufferedElemCube: Send + Sync + Debug {
    fn in_features(&self) -> usize;
    fn out_features(&self) -> usize;
    fn forward_buffered(&self, input: &MatrixBufferHandle, output: &mut MatrixBufferHandle);
    fn backward_buffered(
        &self,
        input: &MatrixBufferHandle,
        output_cache: &MatrixBufferHandle,
        grad_out: &MatrixBufferHandle,
        grad_in: &mut MatrixBufferHandle,
    );
}

pub use sub::Sub;