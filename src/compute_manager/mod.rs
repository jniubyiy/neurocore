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
pub mod device_assignment;
pub mod adaptive_planner;

// Публичные реэкспорты
pub use device::{Device, DeviceDetector, ComputeManager};
pub use executor::Executor;
pub use graph::types::DynamicContext;
pub use dim_change::DynamicTensor;
pub use gpu::GpuExecutor;
pub use matrix_buffer::{MatrixBufferHandle, TempMatrixPool};