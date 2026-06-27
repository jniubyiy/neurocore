// src/compute_manager/mod.rs

pub mod device;
pub mod executor;
pub mod dim_change;
pub mod graph;
pub mod cpu;
pub mod gpu;

// Публичные реэкспорты для удобства пользователей
pub use device::{Device, DeviceDetector, ComputeManager};
pub use executor::Executor;
pub use cpu::CpuExecutor;
pub use graph::model::MixedModel;
pub use graph::types::{DynamicContext, DynamicBatchTensor};
pub use dim_change::DynamicTensor;
pub use gpu::GpuExecutor;