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

// Публичные реэкспорты
pub use device::{Device, DeviceDetector, ComputeManager};
pub use executor::Executor;
pub use graph::types::DynamicContext;
pub use dim_change::DynamicTensor;
pub use gpu::GpuExecutor;
pub use matrix_buffer::{MatrixBufferHandle, TempMatrixPool};

// ============================================================================
// Архитектура v2 — основной путь исполнения.
//
// Модули подключаются безусловно (этап B). Старый путь
// (`MixedModel` + `ComputeExecutor` + `graph::forward`/`graph::backward`)
// удалён на этапе D. Публичные реэкспорты `ComputeExecutor` и
// `ModelPlacement` вместе с ним убраны из этого файла.
// ============================================================================

pub mod jobs_v2;
pub mod operators_v2;
pub mod distributor_v2;
pub mod graph_v2;