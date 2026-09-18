// src/compute_manager/mod.rs
//
// Корневой модуль compute_manager после реорганизации.
//
// Структура:
//   * core/           — базовые типы (Device, DeviceSpec, DynamicTensor,
//                       DynamicContext, Executor);
//   * jobs_v2/        — словарь заданий (Job/JobResult/...);
//   * operators_v2/   — три оператора (Memory / CPU / GPU) и всё
//                       вспомогательное (инфраструктура CPU, GPU, памяти);
//   * distributor_v2/ — SmartDistributor;
//   * graph_v2/       — GraphV2 + observer + bridge.

pub mod core;
pub mod jobs_v2;
pub mod operators_v2;
pub mod distributor_v2;
pub mod graph_v2;

// ---------------------------------------------------------------------------
// Реэкспорты для обратной совместимости.
// ---------------------------------------------------------------------------

pub use core::device::{ComputeManager, Device, DeviceDetector};
pub use core::device_spec::DeviceId;
pub use core::dim_change::DynamicTensor;
pub use core::dynamic_context::{ChunkedContexts, DynamicContext};
pub use core::executor::Executor;

pub use operators_v2::gpu_v2::compute::GpuCompute;
pub use operators_v2::gpu_v2::GpuExecutor;
pub use operators_v2::memory_v2::buffer::{MatrixBufferHandle, TempMatrixPool};
