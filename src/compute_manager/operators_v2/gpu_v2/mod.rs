// src/compute_manager/operators_v2/gpu_v2/mod.rs
//
// GPU-оператор v2 и вся его инфраструктура.
//
// Инфраструктура (переехала из compute_manager/gpu/):
//   * init       — создание Vulkan-контекста;
//   * executor   — GpuExecutor (синхронный);
//   * pipeline   — PipelineCache (общие пайплайны);
//   * compute    — GpuCompute (обёртка над устройством, буферами, пайплайнами);
//   * processor  — forward/backward GPU-сегментов;
//   * shaders    — общие compute-шейдеры.
//
// Оператор:
//   * queue_v2   — выделенный GPU-тред + очередь заданий.

#![allow(dead_code, unused_imports)]

pub mod init;
pub mod executor;
pub mod pipeline;
pub mod compute;
pub mod processor;

pub mod queue_v2;

pub use init::GpuContext;
pub use executor::GpuExecutor;
pub use compute::GpuCompute;
pub use processor::{
    process_forward_gpu_buffered,
    process_backward_gpu_buffered,
};

/// Обнаружить доступные GPU с помощью Vulkan.
/// Возвращает список имён устройств (или None, если Vulkan недоступен).
pub fn detect_gpus() -> Option<Vec<String>> {
    init::enumerate_gpus()
        .map(|gpus| gpus.into_iter().map(|g| g.name).collect())
}
