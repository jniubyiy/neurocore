// src/compute_manager/memory_executor/mod.rs

pub mod types;
pub mod pool;
pub mod executor;
pub mod ssd_cache;
pub mod policy;
pub mod raw_buffer;
pub mod temp_pool;
pub mod data_mover;

// Новые модули для управляемого матричного буфера
pub mod matrix_id;
pub mod matrix_entry;

pub use executor::MemoryExecutor;
pub use executor::MemoryError;
pub use types::{TensorBufferId, BufferLocation, BufferData, TensorBuffer, MemoryDeviceKind};
pub use policy::{BufferPriority, MemoryTier, MemoryPolicy};

// Реэкспорт новых идентификаторов, метаданных и хранилища
pub use matrix_id::MatrixBufferId;
pub use matrix_entry::{MatrixEntry, MatrixStorage};