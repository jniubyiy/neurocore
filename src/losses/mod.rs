// src/losses/mod.rs

use std::any::Any;
use std::fmt::Debug;
use crate::compute_manager::matrix_buffer::MatrixBufferHandle;

pub mod sub;
pub mod square;
pub mod sum_columns;
pub mod log;
pub mod neg;
pub mod mul;
pub mod abs;
pub mod add_scalar;
pub mod log1p;
pub mod abs_diff;
pub mod cross_entropy;

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
pub use square::Square;
pub use sum_columns::SumColumns;
pub use log::Log;
pub use neg::Neg;
pub use mul::Mul;
pub use abs::Abs;
pub use add_scalar::AddScalar;
pub use log1p::Log1p;
pub use abs_diff::AbsDiff;
pub use cross_entropy::CrossEntropyWithLogits;