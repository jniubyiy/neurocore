// src/compute_manager/gpu/mod.rs

// Подмодули GPU-подсистемы
pub mod init;
pub mod executor;
pub mod memory;
pub mod pipeline;
pub mod compute;      // новый модуль: реальные GPU-вычисления
pub mod processor;    // новый модуль: диспетчеризация слоёв на GPU

// Реэкспорт основных типов для удобства использования
pub use init::GpuContext;
pub use executor::GpuExecutor;
pub use memory::GpuTensor;
pub use compute::GpuCompute;
pub use processor::process_forward_gpu;

/// Обнаружить доступные GPU с помощью Vulkan.
/// Возвращает список имён устройств (или None, если Vulkan недоступен).
/// Сохранена для обратной совместимости.
pub fn detect_gpus() -> Option<Vec<String>> {
    init::enumerate_gpus()
        .map(|gpus| gpus.into_iter().map(|g| g.name).collect())
}