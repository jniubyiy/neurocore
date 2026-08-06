// src/compute_manager/memory_executor/raw_buffer.rs

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use vulkano::memory::allocator::MemoryTypeFilter;

use crate::compute_manager::device_spec::DeviceId;
use super::pool::MemoryPool;
use super::types::MemoryDeviceKind;

/// Идентификатор сырого Vulkan-буфера, не связанного напрямую с TensorBuffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RawBufferId(pub usize);

/// Информация о сыром буфере для учёта физической памяти.
#[derive(Debug, Clone)]
pub struct RawBufferInfo {
    pub device_id: DeviceId,
    pub size_bytes: u64,
    pub memory_type: MemoryTypeFilter,
}

/// Реестр сырых Vulkan-буферов, отслеживающий занятую ими физическую память.
pub struct RawBufferRegistry {
    raw_buffers: HashMap<RawBufferId, RawBufferInfo>,
    next_raw_id: AtomicUsize,
}

impl RawBufferRegistry {
    pub fn new() -> Self {
        Self {
            raw_buffers: HashMap::new(),
            next_raw_id: AtomicUsize::new(0),
        }
    }

    /// Регистрирует новый сырой буфер и резервирует память в соответствующем пуле.
    pub fn register(
        &mut self,
        device_id: DeviceId,
        size_bytes: u64,
        memory_type: MemoryTypeFilter,
        pools: &mut HashMap<MemoryDeviceKind, MemoryPool>,
    ) -> RawBufferId {
        let id = RawBufferId(self.next_raw_id.fetch_add(1, Ordering::SeqCst));
        self.raw_buffers.insert(
            id,
            RawBufferInfo {
                device_id,
                size_bytes,
                memory_type,
            },
        );

        let elements = (size_bytes / 4) as usize; // переводим байты в количество f32

        // Определяем, к какому пулу относится буфер
        if memory_type
            .preferred_flags
            .contains(MemoryTypeFilter::PREFER_DEVICE.preferred_flags)
        {
            if let Some(pool) = pools.get_mut(&MemoryDeviceKind::DeviceVram(device_id)) {
                pool.reserve(elements);
            }
        } else if memory_type
            .preferred_flags
            .intersects(MemoryTypeFilter::PREFER_HOST.preferred_flags)
        {
            if let Some(pool) = pools.get_mut(&MemoryDeviceKind::HostRam) {
                pool.reserve(elements);
            }
        }

        id
    }

    /// Снимает регистрацию буфера и освобождает память в пуле.
    pub fn unregister(
        &mut self,
        id: RawBufferId,
        pools: &mut HashMap<MemoryDeviceKind, MemoryPool>,
    ) {
        if let Some(info) = self.raw_buffers.remove(&id) {
            let elements = (info.size_bytes / 4) as usize;
            if info
                .memory_type
                .preferred_flags
                .contains(MemoryTypeFilter::PREFER_DEVICE.preferred_flags)
            {
                if let Some(pool) = pools.get_mut(&MemoryDeviceKind::DeviceVram(info.device_id)) {
                    pool.deallocate(elements);
                }
            } else if info
                .memory_type
                .preferred_flags
                .intersects(MemoryTypeFilter::PREFER_HOST.preferred_flags)
            {
                if let Some(pool) = pools.get_mut(&MemoryDeviceKind::HostRam) {
                    pool.deallocate(elements);
                }
            }
        }
    }

    /// Возвращает информацию о зарегистрированном буфере (например, для отладки).
    pub fn get(&self, id: RawBufferId) -> Option<&RawBufferInfo> {
        self.raw_buffers.get(&id)
    }

    /// Количество зарегистрированных буферов.
    pub fn len(&self) -> usize {
        self.raw_buffers.len()
    }
}