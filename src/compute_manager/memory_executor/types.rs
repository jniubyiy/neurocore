// src/compute_manager/memory_executor/types.rs

use std::path::PathBuf;
use vulkano::buffer::Subbuffer;
use crate::compute_manager::device_spec::DeviceId;
use super::ssd_cache::SsdHandle;
use super::policy::BufferMetadata;
use vulkano::memory::allocator::MemoryTypeFilter;

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

/// Местонахождение данных буфера
#[derive(Debug, Clone)]
pub enum BufferLocation {
    HostRam,
    DeviceVram(DeviceId),
    SsdCache(SsdHandle),
}

/// Внутреннее представление данных буфера
pub enum BufferData {
    HostRam(Vec<f32>),
    DeviceVram(Subbuffer<[f32]>),
    SsdCache(SsdHandle),
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
    /// Метаданные для политики управления памятью
    pub metadata: BufferMetadata,
}

/// Идентификатор «сырого» Vulkan-буфера, не связанного напрямую с TensorBuffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RawBufferId(pub usize);

/// Информация о сыром буфере для учёта физической памяти.
#[derive(Debug, Clone)]
pub struct RawBufferInfo {
    pub device_id: DeviceId,
    pub size_bytes: u64,
    pub memory_type: MemoryTypeFilter,
}