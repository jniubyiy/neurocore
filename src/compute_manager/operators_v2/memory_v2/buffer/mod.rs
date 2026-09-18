// src/compute_manager/operators_v2/memory_v2/buffer/mod.rs
//
// Управляемые матричные буферы. Содержимое переехало из
// compute_manager/matrix_buffer/.

pub mod pool;
pub mod handle;
pub mod guards;
pub mod weak_handle;
pub mod view;
pub mod slice;

pub use pool::TempMatrixPool;
pub use handle::MatrixBufferHandle;
pub use guards::{MatrixReadGuard, MatrixWriteGuard};
pub use weak_handle::WeakMatrixBufferHandle;
pub use view::MatrixBufferView;
pub use slice::MatrixBufferSlice;
