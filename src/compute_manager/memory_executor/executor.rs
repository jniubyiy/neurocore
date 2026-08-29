// src/compute_manager/memory_executor/executor.rs

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};

use super::super::device_spec::{DeviceId, DeviceSpec, DeviceKind};
use super::pool::MemoryPool;
use super::ssd_cache::SsdCacheManager;
use super::types::MemoryDeviceKind;
use super::policy::BufferPriority;
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

    /// Проверяет, можно ли выделить `elements` элементов в указанном пуле памяти.
    pub fn can_allocate(&self, kind: MemoryDeviceKind, elements: usize) -> bool {
        self.pools.get(&kind)
            .map(|p| p.can_allocate(elements))
            .unwrap_or(false)
    }

    // --- Пул временных буферов ---

    pub fn acquire_temp_buffer(
        &mut self,
        kind: MemoryDeviceKind,
        elements: usize,
    ) -> (Subbuffer<[f32]>, RawBufferId) {
        if kind != MemoryDeviceKind::SsdCache {
            if let Some(pool) = self.pools.get(&kind) {
                if !pool.can_allocate(elements) {
                    panic!(
                        "MemoryExecutor::acquire_temp_buffer: insufficient memory for kind {:?}: required {} elements, available {} elements",
                        kind, elements, pool.free_elements()
                    );
                }
            } else {
                panic!(
                    "MemoryExecutor::acquire_temp_buffer: no memory pool registered for kind {:?}",
                    kind
                );
            }
        }

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
        if let Some(pool) = self.pools.get(&location) {
            if !pool.can_allocate(elements) {
                return Err(MemoryError::OutOfMemory(location));
            }
        } else {
            return Err(MemoryError::DeviceNotFound(location));
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
                self.reserve_memory(MemoryDeviceKind::SsdCache, elements)?;
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
                        let elements = handle.elements;
                        self.release_reserved_memory(MemoryDeviceKind::SsdCache, elements);
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

    // ===================================================================
    // БЕСКОПИЙНЫЙ ДОСТУП К CPU-БУФЕРАМ
    // ===================================================================

    pub fn with_cpu_slices<T>(
        &self,
        ids: &[MatrixBufferId],
        f: impl FnOnce(&[&[f32]]) -> T,
    ) -> T {
        let mut slices: Vec<&[f32]> = Vec::with_capacity(ids.len());
        for id in ids {
            let entry = self.matrix_entries.get(id)
                .expect("with_cpu_slices: entry not found");
            match &entry.storage {
                MatrixStorage::Cpu(data) => slices.push(data.as_slice()),
                _ => panic!("with_cpu_slices: buffer is not CPU"),
            }
        }
        f(&slices)
    }

    pub fn with_cpu_slices_mut<T>(
        &mut self,
        ids: &[MatrixBufferId],
        f: impl FnOnce(&mut [&mut [f32]]) -> T,
    ) -> T {
        for i in 0..ids.len() {
            for j in (i + 1)..ids.len() {
                assert_ne!(ids[i], ids[j], "with_cpu_slices_mut: duplicate MatrixBufferId");
            }
        }

        let mut ptrs: Vec<*mut [f32]> = Vec::with_capacity(ids.len());
        for id in ids {
            let entry = self.matrix_entries.get_mut(id)
                .expect("with_cpu_slices_mut: entry not found");
            match &mut entry.storage {
                MatrixStorage::Cpu(data) => {
                    let slice: &mut [f32] = data.as_mut_slice();
                    ptrs.push(slice as *mut [f32]);
                }
                _ => panic!("with_cpu_slices_mut: buffer is not CPU"),
            }
        }

        let mut slices: Vec<&mut [f32]> = ptrs
            .iter()
            .map(|&p| unsafe { &mut *p })
            .collect();

        f(&mut slices)
    }

    pub fn copy_cpu_buffer(&mut self, src_id: MatrixBufferId, dst_id: MatrixBufferId) {
        assert_ne!(src_id, dst_id, "copy_cpu_buffer: source and destination must be different");

        let src_len = {
            let src_entry = self.matrix_entries.get(&src_id)
                .expect("copy_cpu_buffer: source entry not found");
            match &src_entry.storage {
                MatrixStorage::Cpu(data) => data.len(),
                _ => panic!("copy_cpu_buffer: source is not CPU"),
            }
        };
        let dst_len = {
            let dst_entry = self.matrix_entries.get(&dst_id)
                .expect("copy_cpu_buffer: destination entry not found");
            match &dst_entry.storage {
                MatrixStorage::Cpu(data) => data.len(),
                _ => panic!("copy_cpu_buffer: destination is not CPU"),
            }
        };
        assert_eq!(src_len, dst_len, "copy_cpu_buffer: size mismatch");

        let src_ptr = {
            let entry = self.matrix_entries.get(&src_id).unwrap();
            if let MatrixStorage::Cpu(data) = &entry.storage {
                data.as_ptr()
            } else { unreachable!() }
        };
        let dst_ptr = {
            let entry = self.matrix_entries.get_mut(&dst_id).unwrap();
            if let MatrixStorage::Cpu(data) = &mut entry.storage {
                data.as_mut_ptr()
            } else { unreachable!() }
        };
        unsafe {
            std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, src_len);
        }
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

        // Проверяем, достаточно ли памяти в целевом пуле.
        if let Some(pool) = self.pools.get(&target) {
            if !pool.can_allocate(elements) {
                // Попытка вытеснения
                self.evict_to_make_room(target, elements)?;
            }
        } else {
            return Err(MemoryError::DeviceNotFound(target));
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
                    // Важно: staging_buf может быть больше запрошенного размера из-за пула,
                    // поэтому копируем только первые `elements` элементов.
                    write_guard[..elements].copy_from_slice(&data);
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
                self.reserve_memory(MemoryDeviceKind::SsdCache, elements)?;
                MatrixStorage::Ssd(handle)
            }
        };

        self.release_matrix_storage(&old_storage, elements);

        entry.storage = new_storage;
        entry.touch();
        self.matrix_entries.insert(id, entry);
        Ok(())
    }

    /// Освобождает память на целевом устройстве, перемещая наименее используемые
    /// буферы на другие устройства (обычно на HostRam или SsdCache).
    /// Используется при нехватке памяти при миграции.
    pub fn evict_to_make_room(
        &mut self,
        target_kind: MemoryDeviceKind,
        required_elements: usize,
    ) -> Result<(), MemoryError> {
        // Собираем буферы, находящиеся на целевом устройстве.
        let mut candidates: Vec<(MatrixBufferId, Instant, BufferPriority, usize)> = Vec::new();

        for (&id, entry) in &self.matrix_entries {
            if entry.device_kind() == target_kind && !entry.pinned {
                candidates.push((id, entry.last_access, entry.priority, entry.size()));
            }
        }

        // Сортируем: сначала самые старые, затем низкий приоритет.
        candidates.sort_by(|a, b| {
            a.1.cmp(&b.1).then_with(|| priority_rank(a.2).cmp(&priority_rank(b.2)))
        });

        // Пытаемся освободить, перемещая на менее быстрое устройство.
        let mut freed = 0usize;
        for (id, _, priority, size) in candidates {
            if freed >= required_elements {
                break;
            }

            // Определяем, куда переместить: предпочитаем HostRam, затем SsdCache.
            let destination = if self.pools.contains_key(&MemoryDeviceKind::HostRam) {
                MemoryDeviceKind::HostRam
            } else if self.pools.contains_key(&MemoryDeviceKind::SsdCache) {
                MemoryDeviceKind::SsdCache
            } else {
                // Нет доступных устройств для выгрузки.
                break;
            };

            // Не выгружаем высокоприоритетные буферы, если только это не крайняя необходимость.
            if priority == BufferPriority::High && freed < required_elements / 2 {
                continue;
            }

            // Перемещаем буфер.
            if let Err(e) = self.move_matrix_handle(id, destination) {
                // Если переместить не удалось, пропускаем.
                eprintln!("Warning: failed to evict buffer {}: {:?}", id.0, e);
                continue;
            }

            freed += size;
        }

        if freed < required_elements {
            return Err(MemoryError::OutOfMemory(target_kind));
        }
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
                    guard[..elements].to_vec()
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
                self.release_reserved_memory(MemoryDeviceKind::SsdCache, elements);
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
}

fn priority_rank(p: BufferPriority) -> u8 {
    match p {
        BufferPriority::Low => 0,
        BufferPriority::Medium => 1,
        BufferPriority::High => 2,
    }
}