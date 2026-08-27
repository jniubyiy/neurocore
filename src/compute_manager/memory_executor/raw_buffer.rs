// src/compute_manager/memory_executor/raw_buffer.rs

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use vulkano::memory::allocator::MemoryTypeFilter;
use vulkano::memory::MemoryPropertyFlags;

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
    ///
    /// Перед резервированием выполняется проверка наличия свободного места в пуле.
    /// Если памяти недостаточно, генерируется паника с подробным сообщением.
    pub fn register(
        &mut self,
        device_id: DeviceId,
        size_bytes: u64,
        memory_type: MemoryTypeFilter,
        pools: &mut HashMap<MemoryDeviceKind, MemoryPool>,
    ) -> RawBufferId {
        let elements = (size_bytes / 4) as usize; // переводим байты в количество f32

        // Определяем целевой пул на основе флагов памяти.
        let target_pool_kind = memory_kind_from_filter(memory_type, device_id);

        let pool = pools
            .get_mut(&target_pool_kind)
            .unwrap_or_else(|| {
                panic!(
                    "RawBufferRegistry::register: no memory pool registered for kind {:?}",
                    target_pool_kind
                )
            });

        if !pool.can_allocate(elements) {
            panic!(
                "RawBufferRegistry::register: insufficient memory in pool {:?}: required {} elements ({} bytes), available {} elements",
                target_pool_kind,
                elements,
                size_bytes,
                pool.free_elements()
            );
        }

        // Резервируем память в пуле
        pool.reserve(elements);

        // Создаём запись в реестре
        let id = RawBufferId(self.next_raw_id.fetch_add(1, Ordering::SeqCst));
        self.raw_buffers.insert(
            id,
            RawBufferInfo {
                device_id,
                size_bytes,
                memory_type,
            },
        );

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

            let target_pool_kind = memory_kind_from_filter(info.memory_type, info.device_id);

            if let Some(pool) = pools.get_mut(&target_pool_kind) {
                pool.deallocate(elements);
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

/// Определяет тип пула памяти (`MemoryDeviceKind`) на основе флагов `MemoryTypeFilter`.
///
/// Устройство:
/// - если фильтр явно предпочитает или требует `DEVICE_LOCAL`, это VRAM (DeviceVram);
/// - если фильтр явно предпочитает или требует `HOST_VISIBLE`, это HostRam.
///
/// Если ни один из этих флагов не установлен, по умолчанию используется HostRam.
fn memory_kind_from_filter(
    memory_type: MemoryTypeFilter,
    device_id: DeviceId,
) -> MemoryDeviceKind {
    let is_device = memory_type
        .required_flags
        .contains(MemoryPropertyFlags::DEVICE_LOCAL)
        || memory_type
            .preferred_flags
            .contains(MemoryPropertyFlags::DEVICE_LOCAL);

    let is_host = memory_type
        .required_flags
        .contains(MemoryPropertyFlags::HOST_VISIBLE)
        || memory_type
            .preferred_flags
            .contains(MemoryPropertyFlags::HOST_VISIBLE);

    // Если фильтр помечен как DEVICE_LOCAL, считаем это VRAM.
    // В противном случае — HostRam (даже если is_host == false, это безопасное значение по умолчанию).
    if is_device {
        MemoryDeviceKind::DeviceVram(device_id)
    } else {
        MemoryDeviceKind::HostRam
    }
}