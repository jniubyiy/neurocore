// src/compute_manager/memory_executor/executor.rs

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};

use super::super::device_spec::{DeviceId, DeviceSpec, DeviceKind};
use super::pool::MemoryPool;
use super::ssd_cache::SsdCacheManager;
use super::types::MemoryDeviceKind;
use super::policy::{BufferPriority, MemoryPolicy};
use super::raw_buffer::RawBufferRegistry;
use super::temp_pool::TempBufferPool;
use super::data_mover;
use super::matrix_id::MatrixBufferId;
use super::matrix_entry::{MatrixEntry, MatrixStorage};

use crate::compute_manager::gpu::init::GpuContext;
use crate::compute_manager::matrix_buffer::handle::MatrixBufferHandle;

// Публичный реэкспорт для внешних потребителей (GpuCompute)
pub use super::raw_buffer::RawBufferId;

#[derive(Debug)]
pub enum MemoryError {
    OutOfMemory(MemoryDeviceKind),
    DeviceNotFound(MemoryDeviceKind),
    MatrixBufferNotFound(MatrixBufferId),
    SsdError(String),
}

pub struct MemoryExecutor {
    devices: HashMap<DeviceId, DeviceSpec>,
    pools: HashMap<MemoryDeviceKind, MemoryPool>,
    gpu_contexts: HashMap<DeviceId, Arc<GpuContext>>,
    ssd_cache: Option<SsdCacheManager>,
    policy: MemoryPolicy,
    raw_registry: RawBufferRegistry,
    temp_pool: TempBufferPool,
    matrix_entries: HashMap<MatrixBufferId, MatrixEntry>,
    next_matrix_id: AtomicUsize,
    memory_arc: Option<Arc<std::sync::Mutex<MemoryExecutor>>>,
}

impl MemoryExecutor {
    pub fn new() -> Self {
        MemoryExecutor {
            devices: HashMap::new(),
            pools: HashMap::new(),
            gpu_contexts: HashMap::new(),
            ssd_cache: None,
            policy: MemoryPolicy::default(),
            raw_registry: RawBufferRegistry::new(),
            temp_pool: TempBufferPool::new(),
            matrix_entries: HashMap::new(),
            next_matrix_id: AtomicUsize::new(0),
            memory_arc: None,
        }
    }

    pub fn set_self_arc(&mut self, arc: Arc<std::sync::Mutex<MemoryExecutor>>) {
        self.memory_arc = Some(arc);
    }

    // --- Пул временных буферов ---

    pub fn acquire_temp_buffer(
        &mut self,
        kind: MemoryDeviceKind,
        elements: usize,
    ) -> (Subbuffer<[f32]>, RawBufferId) {
        self.temp_pool.acquire(
            kind,
            elements,
            &self.gpu_contexts,
            &mut self.pools,
            &mut self.raw_registry,
        )
    }

    pub fn release_temp_buffer(
        &mut self,
        kind: MemoryDeviceKind,
        buffer: Subbuffer<[f32]>,
        raw_id: RawBufferId,
    ) {
        self.temp_pool.release(kind, buffer, raw_id);
    }

    pub fn cleanup_temp_pools(&mut self, max_age: Duration) {
        self.temp_pool.cleanup(max_age, &mut self.pools, &mut self.raw_registry);
    }

    // --- Регистрация сырых буферов ---

    pub fn register_raw_buffer(
        &mut self,
        device_id: DeviceId,
        size_bytes: u64,
        memory_type: MemoryTypeFilter,
    ) -> RawBufferId {
        self.raw_registry.register(device_id, size_bytes, memory_type, &mut self.pools)
    }

    pub fn unregister_raw_buffer(&mut self, id: RawBufferId) {
        self.raw_registry.unregister(id, &mut self.pools);
    }

    // --- Управление устройствами ---

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

    pub fn set_policy(&mut self, policy: MemoryPolicy) {
        self.policy = policy;
    }

    pub fn current_usage(&self, kind: MemoryDeviceKind) -> usize {
        self.pools
            .get(&kind)
            .map(|p| p.used_elements * 4)
            .unwrap_or(0)
    }

    pub fn gpu_context(&self, device_id: DeviceId) -> Option<&Arc<GpuContext>> {
        self.gpu_contexts.get(&device_id)
    }

    // --- Резервирование памяти ---

    pub fn reserve_memory(&mut self, kind: MemoryDeviceKind, elements: usize) -> Result<(), MemoryError> {
        let pool = self.pools.get_mut(&kind).ok_or(MemoryError::DeviceNotFound(kind))?;
        if pool.can_allocate(elements) {
            pool.reserve(elements);
            Ok(())
        } else {
            Err(MemoryError::OutOfMemory(kind))
        }
    }

    pub fn release_reserved_memory(&mut self, kind: MemoryDeviceKind, elements: usize) {
        if let Some(pool) = self.pools.get_mut(&kind) {
            pool.deallocate(elements);
        }
    }

    // ===================================================================
    // УПРАВЛЯЕМЫЕ МАТРИЧНЫЕ БУФЕРЫ (MatrixBufferHandle)
    // ===================================================================

    pub fn acquire_matrix_handle(
        &mut self,
        rows: usize,
        cols: usize,
        location: MemoryDeviceKind,
        priority: BufferPriority,
    ) -> Result<MatrixBufferHandle, MemoryError> {
        let elements = rows * cols;
        if location != MemoryDeviceKind::SsdCache {
            if let Some(pool) = self.pools.get(&location) {
                if !pool.can_allocate(elements) {
                    return Err(MemoryError::OutOfMemory(location));
                }
            } else {
                return Err(MemoryError::DeviceNotFound(location));
            }
        }

        let storage = match location {
            MemoryDeviceKind::HostRam => {
                self.reserve_memory(MemoryDeviceKind::HostRam, elements)?;
                MatrixStorage::Cpu(vec![0.0f32; elements])
            }
            MemoryDeviceKind::DeviceVram(dev_id) => {
                let (buffer, raw_id) = self.create_raw_gpu_buffer(dev_id, elements)?;
                MatrixStorage::Gpu {
                    buffer,
                    raw_id,
                    device_id: dev_id,
                }
            }
            MemoryDeviceKind::SsdCache => {
                let ssd = self
                    .ssd_cache
                    .as_ref()
                    .ok_or(MemoryError::DeviceNotFound(MemoryDeviceKind::SsdCache))?;
                let handle = ssd.allocate(elements)?;
                let zeros = vec![0.0f32; elements];
                ssd.write(&handle, &zeros)?;
                MatrixStorage::Ssd(handle)
            }
        };

        let id = MatrixBufferId(self.next_matrix_id.fetch_add(1, Ordering::SeqCst));
        let entry = MatrixEntry::new(rows, cols, storage, priority);
        self.matrix_entries.insert(id, entry);

        let arc = self.memory_arc.clone().expect("MemoryExecutor::set_self_arc not called");
        Ok(MatrixBufferHandle::new(id, arc))
    }

    pub fn release_matrix_handle(&mut self, id: MatrixBufferId) {
        if let Some(entry) = self.matrix_entries.get_mut(&id) {
            entry.ref_count = entry.ref_count.saturating_sub(1);
            if entry.ref_count == 0 && !entry.pooled {
                match &entry.storage {
                    MatrixStorage::Cpu(data) => {
                        let elements = data.len();
                        self.release_reserved_memory(MemoryDeviceKind::HostRam, elements);
                    }
                    MatrixStorage::Gpu { raw_id, .. } => {
                        self.raw_registry.unregister(*raw_id, &mut self.pools);
                    }
                    MatrixStorage::Ssd(handle) => {
                        if let Some(ssd) = &self.ssd_cache {
                            let _ = ssd.deallocate(handle);
                        }
                    }
                }
                self.matrix_entries.remove(&id);
            }
        }
    }

    pub fn increment_ref_count(&mut self, id: MatrixBufferId) {
        if let Some(entry) = self.matrix_entries.get_mut(&id) {
            entry.ref_count += 1;
        }
    }

    pub fn mark_pooled(&mut self, id: MatrixBufferId) {
        if let Some(entry) = self.matrix_entries.get_mut(&id) {
            entry.pooled = true;
        }
    }

    pub fn unmark_pooled(&mut self, id: MatrixBufferId) {
        if let Some(entry) = self.matrix_entries.get_mut(&id) {
            entry.pooled = false;
            entry.ref_count += 1;
        }
    }

    pub fn get_matrix_entry(&self, id: MatrixBufferId) -> Option<&MatrixEntry> {
        self.matrix_entries.get(&id)
    }

    pub fn get_matrix_entry_mut(&mut self, id: MatrixBufferId) -> Option<&mut MatrixEntry> {
        self.matrix_entries.get_mut(&id)
    }

    pub fn move_matrix_handle(
        &mut self,
        id: MatrixBufferId,
        target: MemoryDeviceKind,
    ) -> Result<(), MemoryError> {
        let current_kind = {
            let entry = self
                .matrix_entries
                .get(&id)
                .ok_or(MemoryError::MatrixBufferNotFound(id))?;
            entry.device_kind()
        };

        if current_kind == target {
            return Ok(());
        }

        let elements = {
            let entry = self.matrix_entries.get(&id).unwrap();
            entry.rows * entry.cols
        };
        if target != MemoryDeviceKind::SsdCache {
            let pool = self.pools.get(&target)
                .ok_or(MemoryError::DeviceNotFound(target))?;
            if !pool.can_allocate(elements) {
                return Err(MemoryError::OutOfMemory(target));
            }
        }

        let mut entry = self
            .matrix_entries
            .remove(&id)
            .ok_or(MemoryError::MatrixBufferNotFound(id))?;

        let old_storage = entry.storage.clone();
        let new_storage = match target {
            MemoryDeviceKind::HostRam => {
                let data = self.read_matrix_storage_to_vec(&old_storage, elements)?;
                self.reserve_memory(MemoryDeviceKind::HostRam, elements)?;
                MatrixStorage::Cpu(data)
            }
            MemoryDeviceKind::DeviceVram(dev_id) => {
                let data = self.read_matrix_storage_to_vec(&old_storage, elements)?;
                let (buffer, raw_id) = self.create_raw_gpu_buffer(dev_id, elements)?;
                let ctx = self.gpu_contexts.get(&dev_id)
                    .ok_or(MemoryError::DeviceNotFound(MemoryDeviceKind::DeviceVram(dev_id)))?;
                let (staging_buf, staging_raw) = self.temp_pool.acquire(
                    MemoryDeviceKind::HostRam,
                    elements,
                    &self.gpu_contexts,
                    &mut self.pools,
                    &mut self.raw_registry,
                );
                {
                    let mut write_guard = staging_buf.write().map_err(|e| {
                        self.temp_pool.release(MemoryDeviceKind::HostRam, staging_buf.clone(), staging_raw);
                        MemoryError::SsdError(format!("write staging: {}", e))
                    })?;
                    write_guard.copy_from_slice(&data);
                }
                data_mover::copy_buffer_sync(ctx.clone(), staging_buf.clone(), buffer.clone());
                self.temp_pool.release(MemoryDeviceKind::HostRam, staging_buf, staging_raw);
                MatrixStorage::Gpu {
                    buffer,
                    raw_id,
                    device_id: dev_id,
                }
            }
            MemoryDeviceKind::SsdCache => {
                let data = self.read_matrix_storage_to_vec(&old_storage, elements)?;
                let ssd = self.ssd_cache.as_ref()
                    .ok_or(MemoryError::DeviceNotFound(MemoryDeviceKind::SsdCache))?;
                let handle = ssd.allocate(elements)?;
                ssd.write(&handle, &data)?;
                MatrixStorage::Ssd(handle)
            }
        };

        self.release_matrix_storage(&old_storage, elements);

        entry.storage = new_storage;
        entry.touch();
        self.matrix_entries.insert(id, entry);
        Ok(())
    }

    fn read_matrix_storage_to_vec(
        &mut self,
        storage: &MatrixStorage,
        elements: usize,
    ) -> Result<Vec<f32>, MemoryError> {
        match storage {
            MatrixStorage::Cpu(data) => Ok(data.clone()),
            MatrixStorage::Gpu { buffer, device_id, .. } => {
                let ctx = self.gpu_contexts.get(device_id)
                    .ok_or(MemoryError::DeviceNotFound(MemoryDeviceKind::DeviceVram(*device_id)))?;
                let (staging_buf, staging_raw) = self.temp_pool.acquire(
                    MemoryDeviceKind::HostRam,
                    elements,
                    &self.gpu_contexts,
                    &mut self.pools,
                    &mut self.raw_registry,
                );
                data_mover::copy_buffer_sync(ctx.clone(), buffer.clone(), staging_buf.clone());
                let data = {
                    let guard = staging_buf.read().map_err(|e| {
                        self.temp_pool.release(MemoryDeviceKind::HostRam, staging_buf.clone(), staging_raw);
                        MemoryError::SsdError(format!("read staging: {}", e))
                    })?;
                    guard.to_vec()
                };
                self.temp_pool.release(MemoryDeviceKind::HostRam, staging_buf, staging_raw);
                Ok(data)
            }
            MatrixStorage::Ssd(handle) => {
                let ssd = self.ssd_cache.as_ref()
                    .ok_or(MemoryError::DeviceNotFound(MemoryDeviceKind::SsdCache))?;
                ssd.read(handle)
            }
        }
    }

    fn release_matrix_storage(&mut self, storage: &MatrixStorage, elements: usize) {
        match storage {
            MatrixStorage::Cpu(_) => {
                self.release_reserved_memory(MemoryDeviceKind::HostRam, elements);
            }
            MatrixStorage::Gpu { raw_id, .. } => {
                self.raw_registry.unregister(*raw_id, &mut self.pools);
            }
            MatrixStorage::Ssd(handle) => {
                if let Some(ssd) = &self.ssd_cache {
                    let _ = ssd.deallocate(handle);
                }
            }
        }
    }

    fn create_raw_gpu_buffer(
        &mut self,
        device_id: DeviceId,
        elements: usize,
    ) -> Result<(Subbuffer<[f32]>, RawBufferId), MemoryError> {
        let ctx = self
            .gpu_contexts
            .get(&device_id)
            .ok_or(MemoryError::DeviceNotFound(MemoryDeviceKind::DeviceVram(device_id)))?;
        let size_bytes = (elements * std::mem::size_of::<f32>()) as u64;

        let buffer = Buffer::new_unsized(
            ctx.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            size_bytes,
        )
        .map_err(|e| MemoryError::SsdError(format!("Failed to allocate GPU buffer: {}", e)))?;

        let raw_id = self.raw_registry.register(
            device_id,
            size_bytes,
            MemoryTypeFilter::PREFER_DEVICE,
            &mut self.pools,
        );

        Ok((buffer, raw_id))
    }

    pub fn select_matrix_location(
        &self,
        elements: usize,
        preferred: MemoryDeviceKind,
        _priority: BufferPriority,
    ) -> MemoryDeviceKind {
        if self.can_allocate(preferred, elements) {
            return preferred;
        }

        if preferred != MemoryDeviceKind::HostRam
            && self.can_allocate(MemoryDeviceKind::HostRam, elements)
        {
            return MemoryDeviceKind::HostRam;
        }

        if preferred != MemoryDeviceKind::SsdCache
            && self.ssd_cache.is_some()
            && self.can_allocate(MemoryDeviceKind::SsdCache, elements)
        {
            return MemoryDeviceKind::SsdCache;
        }

        preferred
    }

    fn can_allocate(&self, kind: MemoryDeviceKind, elements: usize) -> bool {
        self.pools
            .get(&kind)
            .map(|p| p.can_allocate(elements))
            .unwrap_or(false)
    }
}