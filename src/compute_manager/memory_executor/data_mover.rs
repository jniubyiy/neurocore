// src/compute_manager/memory_executor/data_mover.rs

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::{
    allocator::StandardCommandBufferAllocator,
    AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo,
};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};
use vulkano::sync::{self, GpuFuture};

use crate::compute_manager::gpu::init::GpuContext;

use super::super::device_spec::DeviceId;
use super::pool::MemoryPool;
use super::ssd_cache::SsdCacheManager;
use super::types::{
    BufferData, BufferLocation, MemoryDeviceKind, TensorBuffer, TensorBufferId,
};
use super::raw_buffer::{RawBufferId, RawBufferRegistry};
use super::executor::MemoryError;

pub fn move_buffer_data(
    id: TensorBufferId,
    target: MemoryDeviceKind,
    buffers: &mut HashMap<TensorBufferId, TensorBuffer>,
    pools: &mut HashMap<MemoryDeviceKind, MemoryPool>,
    gpu_contexts: &HashMap<DeviceId, Arc<GpuContext>>,
    ssd_cache: &Option<SsdCacheManager>,
    buffer_to_raw: &mut HashMap<TensorBufferId, RawBufferId>,
    raw_registry: &mut RawBufferRegistry,
) -> Result<(), MemoryError> {
    let buffer = buffers
        .get_mut(&id)
        .ok_or(MemoryError::BufferNotFound(id))?;

    buffer.metadata.touch();
    let current_kind = location_to_kind(&buffer.location);
    if current_kind == target {
        return Ok(());
    }

    let elements = buffer.size_elements;

    if let Some(pool) = pools.get_mut(&current_kind) {
        pool.deallocate(elements);
    }

    let target_pool = pools
        .get_mut(&target)
        .ok_or(MemoryError::DeviceNotFound(target))?;
    if !target_pool.can_allocate(elements) {
        if let Some(pool) = pools.get_mut(&current_kind) {
            pool.allocate(elements).ok();
        }
        return Err(MemoryError::OutOfMemory(target));
    }
    target_pool
        .allocate(elements)
        .map_err(|e| MemoryError::SsdError(e))?;

    // Извлекаем идентификатор GPU, если цель – DeviceVram
    let target_gpu_id = if let MemoryDeviceKind::DeviceVram(id) = target {
        Some(id)
    } else {
        None
    };

    let new_data = match (&buffer.data, &buffer.location, target) {
        (
            BufferData::HostRam(vec),
            BufferLocation::HostRam,
            MemoryDeviceKind::DeviceVram(_),
        ) => {
            let dev_id = target_gpu_id.unwrap();
            let ctx = gpu_contexts
                .get(&dev_id)
                .expect("No GPU context");
            let size_bytes = (elements * std::mem::size_of::<f32>()) as u64;

            let staging_raw = raw_registry.register(
                dev_id,
                size_bytes,
                MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                pools,
            );
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
            .map_err(|e| {
                raw_registry.unregister(staging_raw, pools);
                MemoryError::SsdError(format!("Staging buffer: {}", e))
            })?;
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
            .map_err(|e| {
                raw_registry.unregister(staging_raw, pools);
                MemoryError::SsdError(format!("GPU buffer alloc: {}", e))
            })?;

            copy_buffer_sync(ctx.clone(), staging.clone(), gpu_buf.clone());
            raw_registry.unregister(staging_raw, pools);

            let new_raw_id = raw_registry.register(
                dev_id,
                size_bytes,
                MemoryTypeFilter::PREFER_DEVICE,
                pools,
            );
            buffer_to_raw.insert(id, new_raw_id);

            BufferData::DeviceVram(gpu_buf)
        }
        (
            BufferData::DeviceVram(device_buf),
            BufferLocation::DeviceVram(dev_id),
            MemoryDeviceKind::HostRam,
        ) => {
            let ctx = gpu_contexts
                .get(&dev_id)
                .expect("No GPU context");
            let size_bytes = (elements * std::mem::size_of::<f32>()) as u64;

            let staging_raw = raw_registry.register(
                *dev_id,
                size_bytes,
                MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                pools,
            );
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
            .map_err(|e| {
                raw_registry.unregister(staging_raw, pools);
                MemoryError::SsdError(format!("Staging buffer: {}", e))
            })?;

            copy_buffer_sync(ctx.clone(), device_buf.clone(), staging.clone());
            let data_vec = {
                let guard = staging.read().map_err(|e| {
                    raw_registry.unregister(staging_raw, pools);
                    MemoryError::SsdError(format!("Read staging: {}", e))
                })?;
                let mut v = Vec::with_capacity(guard.len());
                v.extend_from_slice(&guard);
                v
            };
            raw_registry.unregister(staging_raw, pools);

            if let Some(old_raw) = buffer_to_raw.remove(&id) {
                raw_registry.unregister(old_raw, pools);
            }

            BufferData::HostRam(data_vec)
        }
        (
            BufferData::HostRam(vec),
            BufferLocation::HostRam,
            MemoryDeviceKind::SsdCache,
        ) => {
            let ssd = ssd_cache
                .as_ref()
                .expect("SSD cache not registered");
            let handle = ssd.allocate(elements)?;
            ssd.write(&handle, vec)?;
            BufferData::SsdCache(handle)
        }
        (
            BufferData::SsdCache(handle),
            BufferLocation::SsdCache(_),
            MemoryDeviceKind::HostRam,
        ) => {
            let ssd = ssd_cache
                .as_ref()
                .expect("SSD cache not registered");
            let data_vec = ssd.read(handle)?;
            ssd.deallocate(handle)?;
            BufferData::HostRam(data_vec)
        }
        (
            BufferData::DeviceVram(device_buf),
            BufferLocation::DeviceVram(dev_id),
            MemoryDeviceKind::SsdCache,
        ) => {
            let ctx = gpu_contexts
                .get(&dev_id)
                .expect("No GPU context");
            let size_bytes = (elements * std::mem::size_of::<f32>()) as u64;

            let staging_raw = raw_registry.register(
                *dev_id,
                size_bytes,
                MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                pools,
            );
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
            .map_err(|e| {
                raw_registry.unregister(staging_raw, pools);
                MemoryError::SsdError(format!("Staging buffer: {}", e))
            })?;

            copy_buffer_sync(ctx.clone(), device_buf.clone(), staging.clone());
            let data_vec = {
                let guard = staging.read().map_err(|e| {
                    raw_registry.unregister(staging_raw, pools);
                    MemoryError::SsdError(format!("Read staging: {}", e))
                })?;
                let mut v = Vec::with_capacity(guard.len());
                v.extend_from_slice(&guard);
                v
            };
            raw_registry.unregister(staging_raw, pools);

            let ssd = ssd_cache
                .as_ref()
                .expect("SSD cache not registered");
            let handle = ssd.allocate(elements)?;
            ssd.write(&handle, &data_vec)?;

            if let Some(old_raw) = buffer_to_raw.remove(&id) {
                raw_registry.unregister(old_raw, pools);
            }

            BufferData::SsdCache(handle)
        }
        (
            BufferData::SsdCache(handle),
            BufferLocation::SsdCache(_),
            MemoryDeviceKind::DeviceVram(_),
        ) => {
            let dev_id = target_gpu_id.unwrap();
            let ssd = ssd_cache
                .as_ref()
                .expect("SSD cache not registered");
            let data_vec = ssd.read(handle)?;
            ssd.deallocate(handle)?;

            let ctx = gpu_contexts
                .get(&dev_id)
                .expect("No GPU context");
            let size_bytes = (elements * std::mem::size_of::<f32>()) as u64;

            let staging_raw = raw_registry.register(
                dev_id,
                size_bytes,
                MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                pools,
            );
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
            .map_err(|e| {
                raw_registry.unregister(staging_raw, pools);
                MemoryError::SsdError(format!("Staging buffer: {}", e))
            })?;
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
            .map_err(|e| {
                raw_registry.unregister(staging_raw, pools);
                MemoryError::SsdError(format!("GPU buffer alloc: {}", e))
            })?;

            copy_buffer_sync(ctx.clone(), staging.clone(), gpu_buf.clone());
            raw_registry.unregister(staging_raw, pools);

            let new_raw_id = raw_registry.register(
                dev_id,
                size_bytes,
                MemoryTypeFilter::PREFER_DEVICE,
                pools,
            );
            buffer_to_raw.insert(id, new_raw_id);

            BufferData::DeviceVram(gpu_buf)
        }
        _ => {
            if let Some(pool) = pools.get_mut(&current_kind) {
                pool.allocate(elements).ok();
            }
            if let Some(target_pool) = pools.get_mut(&target) {
                target_pool.deallocate(elements);
            }
            return Err(MemoryError::DataNotInLocation(
                id,
                buffer.location.clone(),
            ));
        }
    };

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