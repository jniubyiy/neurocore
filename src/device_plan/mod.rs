// src/device_plan/mod.rs

pub mod plan;
pub mod macros;

pub use plan::{ComputeDevice, DevicePlan, StorageDevice};
// Макрос create_models экспортируется на уровне крейта через #[macro_export],
// поэтому здесь не реэкспортируется, просто используем модуль macros для его определения.