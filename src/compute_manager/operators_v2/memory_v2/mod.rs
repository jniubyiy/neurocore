// src/compute_manager/operators_v2/memory_v2/mod.rs
//
// Memory-оператор v2 и вся его инфраструктура.
//
// Инфраструктура (переехала из compute_manager/memory_executor/):
//   * types        — типы памяти (HostRam / VRAM / SSD);
//   * pool         — учёт занятой памяти в одном устройстве;
//   * executor     — MemoryExecutor (владелец физических данных);
//   * ssd_cache    — SSD-кэш;
//   * policy       — политика вытеснения и продвижения;
//   * raw_buffer   — реестр сырых Vulkan-буферов;
//   * temp_pool    — пул временных Subbuffer'ов;
//   * data_mover   — синхронное копирование между буферами;
//   * matrix_id    — идентификатор управляемого буфера;
//   * matrix_entry — запись реестра (storage + метаданные).
//
// Буферная часть (переехала из compute_manager/matrix_buffer/):
//   * buffer       — MatrixBufferHandle, TempMatrixPool, view, guards и т.д.

pub mod types;
pub mod pool;
pub mod executor;
pub mod ssd_cache;
pub mod policy;
pub mod raw_buffer;
pub mod temp_pool;
pub mod data_mover;

pub mod matrix_id;
pub mod matrix_entry;

pub mod buffer;

pub use executor::MemoryExecutor;
pub use executor::MemoryError;
pub use types::MemoryDeviceKind;
pub use policy::{BufferPriority, MemoryPolicy, MemoryTier};

pub use matrix_id::MatrixBufferId;
pub use matrix_entry::{MatrixEntry, MatrixStorage};
