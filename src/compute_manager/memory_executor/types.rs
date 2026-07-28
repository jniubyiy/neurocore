// src/compute_manager/memory_executor/types.rs

use std::path::PathBuf;
use vulkano::buffer::Subbuffer;
use crate::compute_manager::device_spec::DeviceId;

/// Уникальный идентификатор буфера тензора
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TensorBufferId(pub usize);

/// Тип устройства памяти
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryDeviceKind {
    HostRam,
    DeviceVram(DeviceId),
    SsdCache,
}

/// Местонахождение данных буфера (аналог MemoryDeviceKind, но с путём для SSD)
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BufferLocation {
    HostRam,
    DeviceVram(DeviceId),
    SsdCache(PathBuf),
}

/// Внутреннее представление данных буфера
pub enum BufferData {
    HostRam(Vec<f32>),
    DeviceVram(Subbuffer<[f32]>),
    SsdCache(PathBuf),
    None,
}

/// Дескриптор буфера, управляемый MemoryExecutor
pub struct TensorBuffer {
    pub id: TensorBufferId,
    pub size_elements: usize,
    pub location: BufferLocation,
    pub data: BufferData,
    pub pinned: bool,
    pub use_count: usize,
}