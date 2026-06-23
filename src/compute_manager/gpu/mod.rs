// src/compute_manager/gpu/mod.rs

/// Попытаться обнаружить GPU (CUDA) в системе.
/// Пока возвращает None.
pub fn detect_gpus() -> Option<Vec<String>> {
    // В будущем здесь будет детекция через nvidia-smi или CUDA API
    None
}