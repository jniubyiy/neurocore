// src/compute_manager/mod.rs

pub mod device;
pub mod executor;
pub mod dim_change;
pub mod graph;
pub mod cpu;
pub mod gpu;
pub mod memory_executor;
pub mod matrix_buffer;
pub mod device_spec;
pub mod compute_executor;

// Публичные реэкспорты
pub use device::{Device, DeviceDetector, ComputeManager};
pub use executor::Executor;
pub use graph::types::DynamicContext;
pub use dim_change::DynamicTensor;
pub use gpu::GpuExecutor;
pub use matrix_buffer::{MatrixBufferHandle, TempMatrixPool};
pub use compute_executor::{ComputeExecutor, ModelPlacement};

// ============================================================================
// Новая архитектура v2 (изолирована фичей `v2`).
//
// Ни один из этих модулей не вызывается из существующего кода. Они
// существуют параллельно старому пути (MixedModel / execute / compute_executor)
// и подключаются только на финальном этапе миграции.
//
// Пока фича `v2` выключена, файлы v2 не компилируются и на старый путь
// не влияют.
// ============================================================================

#[cfg(feature = "v2")]
pub mod jobs_v2;

#[cfg(feature = "v2")]
pub mod operators_v2;

#[cfg(feature = "v2")]
pub mod distributor_v2;

#[cfg(feature = "v2")]
pub mod graph_v2;