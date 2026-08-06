// src/compute_manager/memory_executor/temp_pool.rs

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};

use crate::compute_manager::device_spec::DeviceId;
use crate::compute_manager::gpu::init::GpuContext;
use super::pool::MemoryPool;
use super::raw_buffer::RawBufferRegistry;
use super::raw_buffer::RawBufferId;
use super::types::MemoryDeviceKind;

/// Пул временных GPU-буферов для переиспользования.
pub struct TempBufferPool {
    /// Очереди буферов, сгруппированные по типу памяти.
    buffers: HashMap<MemoryDeviceKind, VecDeque<(Subbuffer<[f32]>, RawBufferId, Instant)>>,
}

impl TempBufferPool {
    pub fn new() -> Self {
        Self {
            buffers: HashMap::new(),
        }
    }

    /// Получить временный буфер заданного размера (в элементах f32) из пула или создать новый.
    /// Регистрирует буфер в `raw_registry` и обновляет `pools`.
    pub fn acquire(
        &mut self,
        kind: MemoryDeviceKind,
        elements: usize,
        gpu_contexts: &HashMap<DeviceId, Arc<GpuContext>>,
        pools: &mut HashMap<MemoryDeviceKind, MemoryPool>,
        raw_registry: &mut RawBufferRegistry,
    ) -> (Subbuffer<[f32]>, RawBufferId) {
        let device_id = match kind {
            MemoryDeviceKind::DeviceVram(id) => id,
            _ => panic!("TempBufferPool only supports DeviceVram"),
        };

        let queue = self.buffers.entry(kind).or_insert_with(VecDeque::new);

        // Ищем подходящий буфер
        if let Some(pos) = queue.iter().position(|(buf, _, _)| buf.len() >= elements as u64) {
            let (buf, raw_id, _) = queue.remove(pos).unwrap();
            return (buf, raw_id);
        }

        // Создаём новый буфер
        let ctx = gpu_contexts
            .get(&device_id)
            .expect("GPU context not found");
        let size_bytes = (elements * std::mem::size_of::<f32>()) as u64;
        let buffer = Buffer::new_unsized(
            ctx.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER
                    | BufferUsage::TRANSFER_SRC
                    | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            size_bytes,
        )
        .expect("Failed to create temp GPU buffer");

        // Регистрируем как сырой буфер
        let raw_id =
            raw_registry.register(device_id, size_bytes, MemoryTypeFilter::PREFER_DEVICE, pools);

        (buffer, raw_id)
    }

    /// Возвращает временный буфер обратно в пул.
    pub fn release(
        &mut self,
        kind: MemoryDeviceKind,
        buffer: Subbuffer<[f32]>,
        raw_id: RawBufferId,
    ) {
        let queue = self.buffers.entry(kind).or_insert_with(VecDeque::new);
        queue.push_back((buffer, raw_id, Instant::now()));
    }

    /// Удаляет из пула буферы, которые не использовались дольше `max_age`.
    /// Освобождает память в пулах и разрегистрирует raw-буферы.
    pub fn cleanup(
        &mut self,
        max_age: Duration,
        pools: &mut HashMap<MemoryDeviceKind, MemoryPool>,
        raw_registry: &mut RawBufferRegistry,
    ) {
        let now = Instant::now();
        let mut ids_to_remove = Vec::new();

        for queue in self.buffers.values_mut() {
            while let Some((_, raw_id, last_used)) = queue.front() {
                if now.duration_since(*last_used) >= max_age {
                    ids_to_remove.push(*raw_id);
                    queue.pop_front();
                } else {
                    break;
                }
            }
        }

        // Удаляем raw-буферы вне итерации по очереди
        for raw_id in ids_to_remove {
            raw_registry.unregister(raw_id, pools);
        }
    }

    /// Возвращает общее количество буферов в пуле.
    pub fn total_buffers(&self) -> usize {
        self.buffers.values().map(|q| q.len()).sum()
    }
}