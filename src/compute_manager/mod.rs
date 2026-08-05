// src/compute_manager/mod.rs

pub mod device;
pub mod executor;
pub mod dim_change;
pub mod graph;
pub mod cpu;
pub mod gpu;
pub mod memory_executor;
pub mod device_spec;
pub mod device_assignment;   // назначение устройств для сегментов
pub mod device_tensor;       // абстракция тензора над устройствами

// Публичные реэкспорты для удобства пользователей.
// MixedModel больше не экспортируется – модель создаётся только через TrainingPlan.
pub use device::{Device, DeviceDetector, ComputeManager};
pub use executor::Executor;
pub use graph::types::{DynamicContext, DynamicBatchTensor};
pub use dim_change::DynamicTensor;
pub use gpu::GpuExecutor;