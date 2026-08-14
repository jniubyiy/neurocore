// src/compute_manager/gpu/mod.rs

// Подмодули GPU-подсистемы
pub mod init;
pub mod executor;
pub mod memory;
pub mod pipeline;
pub mod compute;
pub mod processor;
pub mod param_store;    // <-- добавлено

// Реэкспорт основных типов для удобства использования
pub use init::GpuContext;
pub use executor::GpuExecutor;
pub use memory::GpuTensor;
pub use compute::GpuCompute;
pub use processor::{
    process_forward_gpu_buffered,
    process_backward_gpu_buffered,
};
pub use param_store::GpuParamStore;   // <-- добавлено

/// Обнаружить доступные GPU с помощью Vulkan.
/// Возвращает список имён устройств (или None, если Vulkan недоступен).
/// Сохранена для обратной совместимости.
pub fn detect_gpus() -> Option<Vec<String>> {
    init::enumerate_gpus()
        .map(|gpus| gpus.into_iter().map(|g| g.name).collect())
}