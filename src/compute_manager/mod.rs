// src/compute_manager/mod.rs

pub mod device;
pub mod executor;
pub mod dim_change;
pub mod graph;
pub mod cpu;
pub mod gpu;
pub mod memory_executor;
pub mod device_spec;
pub mod device_plan;
pub mod logger;          // <-- добавлено
pub mod diagnostics;    // <-- добавлено

// Публичные реэкспорты для удобства пользователей
pub use device::{Device, DeviceDetector, ComputeManager};
pub use executor::Executor;
// pub use cpu::CpuExecutor;   // удаляем, т.к. CpuExecutor приватный
pub use graph::model::MixedModel;
pub use graph::types::{DynamicContext, DynamicBatchTensor};
pub use dim_change::DynamicTensor;
pub use gpu::GpuExecutor;
pub use device_plan::DevicePlan;