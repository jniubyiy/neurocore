// src/compute_manager/matrix_buffer/mod.rs

pub mod pool;
pub mod handle;
pub mod guards;
pub mod weak_handle;

pub use pool::TempMatrixPool;
pub use handle::MatrixBufferHandle;
pub use guards::{MatrixReadGuard, MatrixWriteGuard};
pub use weak_handle::WeakMatrixBufferHandle;