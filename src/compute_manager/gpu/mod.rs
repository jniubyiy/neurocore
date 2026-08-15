// src/compute_manager/gpu/mod.rs

pub mod init;
pub mod executor;
pub mod pipeline;
pub mod compute;
pub mod processor;

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