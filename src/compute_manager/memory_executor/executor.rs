// src/compute_manager/memory_executor/executor.rs

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::{
    allocator::StandardCommandBufferAllocator,
    AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo,
};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};
use vulkano::sync::{self, GpuFuture};

use super::super::device_spec::{DeviceId, DeviceSpec, DeviceKind};
use super::pool::MemoryPool;
use super::ssd_cache::SsdCacheManager;
use super::types::{
    BufferData, BufferLocation, MemoryDeviceKind, TensorBuffer, TensorBufferId,
};

use crate::compute_manager::gpu::init::GpuContext;

#[derive(Debug)]
pub enum MemoryError {
    OutOfMemory(MemoryDeviceKind),
    DeviceNotFound(MemoryDeviceKind),
    BufferNotFound(TensorBufferId),
    DataNotInLocation(TensorBufferId, BufferLocation),
    SsdError(String),
}

pub struct MemoryExecutor {
    devices: HashMap<DeviceId, DeviceSpec>,
    pools: HashMap<MemoryDeviceKind, MemoryPool>,
    buffers: HashMap<TensorBufferId, TensorBuffer>,
    next_buffer_id: AtomicUsize,
    gpu_contexts: HashMap<DeviceId, Arc<GpuContext>>,
    ssd_cache: Option<SsdCacheManager>,
}

impl MemoryExecutor {
    pub fn new() -> Self {
        MemoryExecutor {
            devices: HashMap::new(),
            pools: HashMap::new(),
            buffers: HashMap::new(),
            next_buffer_id: AtomicUsize::new(0),
            gpu_contexts: HashMap::new(),
            ssd_cache: None,
        }
    }

    pub fn register_compute_device(
        &mut self,
        spec: DeviceSpec,
        gpu_context: Option<Arc<GpuContext>>,
    ) {
        let id = spec.id;
        let max_bytes = spec.limits.max_memory_mb * 1024 * 1024;
        match spec.kind {
            DeviceKind::Cpu => {
                if !self.pools.contains_key(&MemoryDeviceKind::HostRam) {
                    self.pools
                        .insert(MemoryDeviceKind::HostRam, MemoryPool::new(max_bytes));
                }
            }
            DeviceKind::Gpu => {
                let kind = MemoryDeviceKind::DeviceVram(id);
                self.pools.insert(kind, MemoryPool::new(max_bytes));
                if let Some(ctx) = gpu_context {
                    self.gpu_contexts.insert(id, ctx);
                }
            }
        }
        self.devices.insert(id, spec);
    }

    /// Регистрирует SSD-кэш с заданной директорией и максимальным размером.
    pub fn register_ssd_cache(
        &mut self,
        path: PathBuf,
        max_bytes: u64,
    ) -> Result<(), MemoryError> {
        let manager = SsdCacheManager::new(path, max_bytes)?;
        self.pools.insert(
            MemoryDeviceKind::SsdCache,
            MemoryPool::new(max_bytes),
        );
        self.ssd_cache = Some(manager);
        Ok(())
    }

    /// Возвращает текущее использование памяти (в байтах) для заданного типа памяти.
    pub fn current_usage(&self, kind: MemoryDeviceKind) -> usize {
        self.pools
            .get(&kind)
            .map(|p| p.used_elements * 4)   // каждый f32 = 4 байта
            .unwrap_or(0)
    }

    pub fn allocate(
        &mut self,
        location: MemoryDeviceKind,
        elements: usize,
    ) -> Result<TensorBufferId, MemoryError> {
        let pool = self
            .pools
            .get_mut(&location)
            .ok_or(MemoryError::DeviceNotFound(location))?;
        if !pool.can_allocate(elements) {
            return Err(MemoryError::OutOfMemory(location));
        }
        pool.allocate(elements)
            .map_err(|e| MemoryError::SsdError(e))?;

        let (data, buffer_location) = match location {
            MemoryDeviceKind::HostRam => {
                (BufferData::HostRam(vec![0.0f32; elements]), BufferLocation::HostRam)
            }
            MemoryDeviceKind::DeviceVram(dev_id) => {
                let ctx = self
                    .gpu_contexts
                    .get(&dev_id)
                    .expect("GPU context not registered");
                let size_bytes = (elements * std::mem::size_of::<f32>()) as u64;
                let buffer = Buffer::new_unsized(
                    ctx.memory_allocator.clone(),
                    BufferCreateInfo {
                        usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC,
                        ..Default::default()
                    },
                    AllocationCreateInfo {
                        memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                        ..Default::default()
                    },
                    size_bytes,
                )
                .map_err(|e| MemoryError::SsdError(format!("Failed to allocate GPU buffer: {}", e)))?;
                (
                    BufferData::DeviceVram(buffer),
                    BufferLocation::DeviceVram(dev_id),
                )
            }
            MemoryDeviceKind::SsdCache => {
                let ssd = self
                    .ssd_cache
                    .as_ref()
                    .ok_or(MemoryError::DeviceNotFound(MemoryDeviceKind::SsdCache))?;
                let handle = ssd.allocate(elements)?;
                (
                    BufferData::SsdCache(handle.clone()),
                    BufferLocation::SsdCache(handle),
                )
            }
        };

        let id = TensorBufferId(self.next_buffer_id.fetch_add(1, Ordering::SeqCst));
        let buffer = TensorBuffer {
            id,
            size_elements: elements,
            location: buffer_location,
            data,
            pinned: false,
            use_count: 0,
        };
        self.buffers.insert(id, buffer);
        Ok(id)
    }

    pub fn move_buffer(
        &mut self,
        id: TensorBufferId,
        target: MemoryDeviceKind,
    ) -> Result<(), MemoryError> {
        let buffer = self
            .buffers
            .get_mut(&id)
            .ok_or(MemoryError::BufferNotFound(id))?;
        let current_kind = location_to_kind(&buffer.location);
        if current_kind == target {
            return Ok(());
        }

        let elements = buffer.size_elements;

        // освобождаем исходный пул
        if let Some(pool) = self.pools.get_mut(&current_kind) {
            pool.deallocate(elements);
        }

        // резервируем целевой пул
        let target_pool = self
            .pools
            .get_mut(&target)
            .ok_or(MemoryError::DeviceNotFound(target))?;
        if !target_pool.can_allocate(elements) {
            // возвращаем память обратно
            if let Some(pool) = self.pools.get_mut(&current_kind) {
                pool.allocate(elements).ok();
            }
            return Err(MemoryError::OutOfMemory(target));
        }
        target_pool
            .allocate(elements)
            .map_err(|e| MemoryError::SsdError(e))?;

        // выполняем фактическое перемещение данных
        let new_data = match (&buffer.data, &buffer.location, target) {
            (BufferData::HostRam(vec), BufferLocation::HostRam, MemoryDeviceKind::DeviceVram(dev_id)) => {
                let ctx = self.gpu_contexts.get(&dev_id).expect("No GPU context");
                let size_bytes = (elements * std::mem::size_of::<f32>()) as u64;
                let gpu_buf = Buffer::new_unsized(
                    ctx.memory_allocator.clone(),
                    BufferCreateInfo {
                        usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
                        ..Default::default()
                    },
                    AllocationCreateInfo {
                        memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                        ..Default::default()
                    },
                    size_bytes,
                )
                .map_err(|e| MemoryError::SsdError(format!("GPU buffer alloc: {}", e)))?;
                let staging = Buffer::from_iter(
                    ctx.memory_allocator.clone(),
                    BufferCreateInfo {
                        usage: BufferUsage::TRANSFER_SRC,
                        ..Default::default()
                    },
                    AllocationCreateInfo {
                        memory_type_filter: MemoryTypeFilter::PREFER_HOST
                            | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                        ..Default::default()
                    },
                    vec.iter().copied(),
                )
                .map_err(|e| MemoryError::SsdError(format!("Staging buffer: {}", e)))?;
                copy_buffer_sync(ctx.clone(), staging, gpu_buf.clone());
                BufferData::DeviceVram(gpu_buf)
            }
            (BufferData::DeviceVram(device_buf), BufferLocation::DeviceVram(dev_id), MemoryDeviceKind::HostRam) => {
                let ctx = self.gpu_contexts.get(&dev_id).expect("No GPU context");
                let size_bytes = (elements * std::mem::size_of::<f32>()) as u64;
                let staging = Buffer::new_unsized(
                    ctx.memory_allocator.clone(),
                    BufferCreateInfo {
                        usage: BufferUsage::TRANSFER_DST,
                        ..Default::default()
                    },
                    AllocationCreateInfo {
                        memory_type_filter: MemoryTypeFilter::PREFER_HOST
                            | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                        ..Default::default()
                    },
                    size_bytes,
                )
                .map_err(|e| MemoryError::SsdError(format!("Staging buffer: {}", e)))?;
                copy_buffer_sync(ctx.clone(), device_buf.clone(), staging.clone());
                let data_vec = {
                    let guard = staging
                        .read()
                        .map_err(|e| MemoryError::SsdError(format!("Read staging: {}", e)))?;
                    let mut v = Vec::with_capacity(guard.len());
                    v.extend_from_slice(&guard);
                    v
                };
                BufferData::HostRam(data_vec)
            }
            (BufferData::HostRam(vec), BufferLocation::HostRam, MemoryDeviceKind::SsdCache) => {
                let ssd = self.ssd_cache.as_ref().expect("SSD cache not registered");
                let handle = ssd.allocate(elements)?;
                ssd.write(&handle, vec)?;
                BufferData::SsdCache(handle)
            }
            (BufferData::SsdCache(handle), BufferLocation::SsdCache(_), MemoryDeviceKind::HostRam) => {
                let ssd = self.ssd_cache.as_ref().expect("SSD cache not registered");
                let data_vec = ssd.read(handle)?;
                ssd.deallocate(handle)?;
                BufferData::HostRam(data_vec)
            }
            (BufferData::DeviceVram(device_buf), BufferLocation::DeviceVram(dev_id), MemoryDeviceKind::SsdCache) => {
                let ctx = self.gpu_contexts.get(&dev_id).expect("No GPU context");
                let size_bytes = (elements * std::mem::size_of::<f32>()) as u64;
                let staging = Buffer::new_unsized(
                    ctx.memory_allocator.clone(),
                    BufferCreateInfo {
                        usage: BufferUsage::TRANSFER_DST,
                        ..Default::default()
                    },
                    AllocationCreateInfo {
                        memory_type_filter: MemoryTypeFilter::PREFER_HOST
                            | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                        ..Default::default()
                    },
                    size_bytes,
                )
                .map_err(|e| MemoryError::SsdError(format!("Staging buffer: {}", e)))?;
                copy_buffer_sync(ctx.clone(), device_buf.clone(), staging.clone());
                let data_vec = {
                    let guard = staging
                        .read()
                        .map_err(|e| MemoryError::SsdError(format!("Read staging: {}", e)))?;
                    let mut v = Vec::with_capacity(guard.len());
                    v.extend_from_slice(&guard);
                    v
                };
                let ssd = self.ssd_cache.as_ref().expect("SSD cache not registered");
                let handle = ssd.allocate(elements)?;
                ssd.write(&handle, &data_vec)?;
                BufferData::SsdCache(handle)
            }
            (BufferData::SsdCache(handle), BufferLocation::SsdCache(_), MemoryDeviceKind::DeviceVram(dev_id)) => {
                let ssd = self.ssd_cache.as_ref().expect("SSD cache not registered");
                let data_vec = ssd.read(handle)?;
                ssd.deallocate(handle)?;
                let ctx = self.gpu_contexts.get(&dev_id).expect("No GPU context");
                let size_bytes = (elements * std::mem::size_of::<f32>()) as u64;
                let gpu_buf = Buffer::new_unsized(
                    ctx.memory_allocator.clone(),
                    BufferCreateInfo {
                        usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
                        ..Default::default()
                    },
                    AllocationCreateInfo {
                        memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                        ..Default::default()
                    },
                    size_bytes,
                )
                .map_err(|e| MemoryError::SsdError(format!("GPU buffer alloc: {}", e)))?;
                let staging = Buffer::from_iter(
                    ctx.memory_allocator.clone(),
                    BufferCreateInfo {
                        usage: BufferUsage::TRANSFER_SRC,
                        ..Default::default()
                    },
                    AllocationCreateInfo {
                        memory_type_filter: MemoryTypeFilter::PREFER_HOST
                            | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                        ..Default::default()
                    },
                    data_vec.iter().copied(),
                )
                .map_err(|e| MemoryError::SsdError(format!("Staging buffer: {}", e)))?;
                copy_buffer_sync(ctx.clone(), staging, gpu_buf.clone());
                BufferData::DeviceVram(gpu_buf)
            }
            _ => {
                // откат: освобождаем целевой пул и возвращаем исходный
                if let Some(pool) = self.pools.get_mut(&current_kind) {
                    pool.allocate(elements).ok();
                }
                if let Some(target_pool) = self.pools.get_mut(&target) {
                    target_pool.deallocate(elements);
                }
                return Err(MemoryError::DataNotInLocation(id, buffer.location.clone()));
            }
        };

        // обновляем местонахождение
        buffer.data = new_data;
        buffer.location = match target {
            MemoryDeviceKind::HostRam => BufferLocation::HostRam,
            MemoryDeviceKind::DeviceVram(dev_id) => BufferLocation::DeviceVram(dev_id),
            MemoryDeviceKind::SsdCache => {
                if let BufferData::SsdCache(ref handle) = buffer.data {
                    BufferLocation::SsdCache(handle.clone())
                } else {
                    BufferLocation::SsdCache(super::ssd_cache::SsdHandle {
                        file_path: PathBuf::new(),
                        elements: 0,
                    })
                }
            }
        };

        Ok(())
    }

    pub fn resolve_buffer(
        &mut self,
        id: TensorBufferId,
        target: MemoryDeviceKind,
    ) -> Result<ResolvedBuffer<'_>, MemoryError> {
        self.move_buffer(id, target)?;
        let buffer = self.buffers.get_mut(&id).unwrap();
        buffer.use_count += 1;
        Ok(ResolvedBuffer {
            buffer: buffer as *mut TensorBuffer,
            _owner: std::marker::PhantomData,
        })
    }

    pub fn release_buffer(&mut self, id: TensorBufferId) {
        if let Some(buffer) = self.buffers.get_mut(&id) {
            buffer.use_count = buffer.use_count.saturating_sub(1);
        }
    }

    pub fn deallocate_buffer(&mut self, id: TensorBufferId) -> Result<(), MemoryError> {
        let buffer = self.buffers.remove(&id).ok_or(MemoryError::BufferNotFound(id))?;
        let kind = location_to_kind(&buffer.location);
        if let Some(pool) = self.pools.get_mut(&kind) {
            pool.deallocate(buffer.size_elements);
        }
        if let BufferData::SsdCache(handle) = buffer.data {
            if let Some(ssd) = &self.ssd_cache {
                ssd.deallocate(&handle)?;
            }
        }
        Ok(())
    }
}

// Вспомогательные функции
fn location_to_kind(loc: &BufferLocation) -> MemoryDeviceKind {
    match loc {
        BufferLocation::HostRam => MemoryDeviceKind::HostRam,
        BufferLocation::DeviceVram(id) => MemoryDeviceKind::DeviceVram(*id),
        BufferLocation::SsdCache(_) => MemoryDeviceKind::SsdCache,
    }
}

fn copy_buffer_sync(
    ctx: Arc<GpuContext>,
    src: Subbuffer<[f32]>,
    dst: Subbuffer<[f32]>,
) {
    let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
        ctx.device.clone(),
        Default::default(),
    ));
    let mut builder = AutoCommandBufferBuilder::primary(
        command_buffer_allocator,
        ctx.queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();
    builder
        .copy_buffer(CopyBufferInfo::buffers(src, dst))
        .unwrap();
    let cb = builder.build().unwrap();
    let future = sync::now(ctx.device.clone())
        .then_execute(ctx.queue.clone(), cb)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap();
    future.wait(None).unwrap();
}

pub struct ResolvedBuffer<'a> {
    buffer: *mut TensorBuffer,
    _owner: std::marker::PhantomData<&'a mut MemoryExecutor>,
}

impl<'a> ResolvedBuffer<'a> {
    pub fn as_host_slice(&self) -> &[f32] {
        let buf = unsafe { &*self.buffer };
        match &buf.data {
            BufferData::HostRam(data) => data.as_slice(),
            _ => panic!("Buffer not in HostRam"),
        }
    }
    pub fn as_host_slice_mut(&mut self) -> &mut [f32] {
        let buf = unsafe { &mut *self.buffer };
        match &mut buf.data {
            BufferData::HostRam(data) => data.as_mut_slice(),
            _ => panic!("Buffer not in HostRam"),
        }
    }
    pub fn as_device_buffer(&self) -> &Subbuffer<[f32]> {
        let buf = unsafe { &*self.buffer };
        match &buf.data {
            BufferData::DeviceVram(buffer) => buffer,
            _ => panic!("Buffer not in DeviceVram"),
        }
    }
}