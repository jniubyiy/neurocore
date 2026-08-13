// src/compute_manager/memory_executor/executor.rs

use std::collections::HashMap;
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};

use super::super::device_spec::{DeviceId, DeviceSpec, DeviceKind};
use super::pool::MemoryPool;
use super::ssd_cache::SsdCacheManager;
use super::types::{
    BufferData, BufferLocation, MemoryDeviceKind, TensorBuffer, TensorBufferId,
};
use super::policy::{BufferMetadata, BufferPriority, MemoryPolicy, MemoryTier};
use super::raw_buffer::{RawBufferRegistry};
use super::temp_pool::TempBufferPool;
use super::data_mover;
use super::matrix_id::MatrixBufferId;
use super::matrix_registry::MatrixBufferInfo;

use crate::compute_manager::gpu::init::GpuContext;
use crate::compute_manager::matrix_buffer::buffer::{BufferStorage, MatrixBuffer};

// Публичный реэкспорт для внешних потребителей (GpuCompute, GpuParamStore)
pub use super::raw_buffer::RawBufferId;

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
    policy: MemoryPolicy,
    tick_counter: usize,
    upcoming_ids: HashSet<TensorBufferId>,

    // Связь TensorBufferId -> RawBufferId для буферов DeviceVram
    buffer_to_raw: HashMap<TensorBufferId, RawBufferId>,

    // Компоненты управления памятью
    raw_registry: RawBufferRegistry,
    temp_pool: TempBufferPool,

    // Закреплённые (pinned) буферы, которые не участвуют в автоматическом вытеснении
    pinned_buffers: HashSet<TensorBufferId>,

    // Новые поля для управляемых матричных буферов
    matrix_buffers: HashMap<MatrixBufferId, MatrixBufferInfo>,
    next_matrix_id: AtomicUsize,

    // Ссылка на самого себя, обёрнутого в Arc<Mutex<...>>.
    // Устанавливается после создания через `set_self_arc`.
    memory_arc: Option<Arc<std::sync::Mutex<MemoryExecutor>>>,
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
            policy: MemoryPolicy::default(),
            tick_counter: 0,
            upcoming_ids: HashSet::new(),
            buffer_to_raw: HashMap::new(),
            raw_registry: RawBufferRegistry::new(),
            temp_pool: TempBufferPool::new(),
            pinned_buffers: HashSet::new(),
            matrix_buffers: HashMap::new(),
            next_matrix_id: AtomicUsize::new(0),
            memory_arc: None,
        }
    }

    /// Устанавливает ссылку на самого себя, обёрнутого в `Arc<Mutex<MemoryExecutor>>`.
    /// Этот метод должен быть вызван сразу после создания `Arc<Mutex<MemoryExecutor>>`.
    pub fn set_self_arc(&mut self, arc: Arc<std::sync::Mutex<MemoryExecutor>>) {
        self.memory_arc = Some(arc);
    }

    // --- Пул временных буферов (делегирование) ---

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

    // --- Регистрация сырых буферов (делегирование) ---

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

    /// Предоставляет доступ к GPU-контексту по идентификатору устройства.
    pub fn gpu_context(&self, device_id: DeviceId) -> Option<&Arc<GpuContext>> {
        self.gpu_contexts.get(&device_id)
    }

    fn get_usage_ratios(&self) -> (f32, f32) {
        let vram_used = self.pools
            .iter()
            .filter(|(k, _)| matches!(k, MemoryDeviceKind::DeviceVram(_)))
            .map(|(_, p)| p.used_elements)
            .sum::<usize>();
        let vram_total = self.pools
            .iter()
            .filter(|(k, _)| matches!(k, MemoryDeviceKind::DeviceVram(_)))
            .map(|(_, p)| p.max_elements)
            .sum::<usize>();

        let ram_used = self.pools
            .get(&MemoryDeviceKind::HostRam)
            .map(|p| p.used_elements)
            .unwrap_or(0);
        let ram_total = self.pools
            .get(&MemoryDeviceKind::HostRam)
            .map(|p| p.max_elements)
            .unwrap_or(1);

        let vram_ratio = if vram_total > 0 { vram_used as f32 / vram_total as f32 } else { 0.0 };
        let ram_ratio = if ram_total > 0 { ram_used as f32 / ram_total as f32 } else { 0.0 };
        (vram_ratio, ram_ratio)
    }

    fn ram_usage_ratio(&self) -> f32 {
        let (_, ram_ratio) = self.get_usage_ratios();
        ram_ratio
    }

    // --- Выделение и освобождение тензоров ---

    pub fn allocate(
        &mut self,
        location: MemoryDeviceKind,
        elements: usize,
        priority: BufferPriority,
    ) -> Result<TensorBufferId, MemoryError> {
        let need_evict = {
            let pool = self.pools.get(&location).ok_or(MemoryError::DeviceNotFound(location))?;
            !pool.can_allocate(elements) && location != MemoryDeviceKind::SsdCache
        };
        if need_evict {
            self.evict_to_fit(location, elements)?;
        }

        let pool = self
            .pools
            .get_mut(&location)
            .ok_or(MemoryError::DeviceNotFound(location))?;

        if !pool.can_allocate(elements) {
            return Err(MemoryError::OutOfMemory(location));
        }
        pool.allocate(elements)
            .map_err(|e| MemoryError::SsdError(e))?;

        let (data, buffer_location, raw_id) = match location {
            MemoryDeviceKind::HostRam => {
                (BufferData::HostRam(vec![0.0f32; elements]), BufferLocation::HostRam, None)
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
                let raw_id = self.raw_registry.register(dev_id, size_bytes, MemoryTypeFilter::PREFER_DEVICE, &mut self.pools);
                (
                    BufferData::DeviceVram(buffer),
                    BufferLocation::DeviceVram(dev_id),
                    Some(raw_id),
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
                    None,
                )
            }
        };

        let id = TensorBufferId(self.next_buffer_id.fetch_add(1, Ordering::SeqCst));
        if let Some(raw_id) = raw_id {
            self.buffer_to_raw.insert(id, raw_id);
        }
        let metadata = BufferMetadata::new(elements, priority);
        let buffer = TensorBuffer {
            id,
            size_elements: elements,
            location: buffer_location,
            data,
            pinned: false,
            use_count: 0,
            metadata,
            is_temp: false,
        };
        self.buffers.insert(id, buffer);
        Ok(id)
    }

    pub fn allocate_pinned(
        &mut self,
        location: MemoryDeviceKind,
        elements: usize,
        priority: BufferPriority,
    ) -> Result<TensorBufferId, MemoryError> {
        let id = self.allocate(location, elements, priority)?;
        if let Some(buffer) = self.buffers.get_mut(&id) {
            buffer.pinned = true;
        }
        self.pinned_buffers.insert(id);
        Ok(id)
    }

    pub fn deallocate_pinned(&mut self, id: TensorBufferId) -> Result<(), MemoryError> {
        self.pinned_buffers.remove(&id);
        self.deallocate_buffer(id)
    }

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

    // --- Перемещение буфера ---

    pub fn move_buffer(
        &mut self,
        id: TensorBufferId,
        target: MemoryDeviceKind,
    ) -> Result<(), MemoryError> {
        data_mover::move_buffer_data(
            id,
            target,
            &mut self.buffers,
            &mut self.pools,
            &self.gpu_contexts,
            &self.ssd_cache,
            &mut self.buffer_to_raw,
            &mut self.raw_registry,
            &mut self.temp_pool,
        )
    }

    // --- Вытеснение ---

    fn evict_to_fit(
        &mut self,
        kind: MemoryDeviceKind,
        required_elements: usize,
    ) -> Result<(), MemoryError> {
        if kind == MemoryDeviceKind::SsdCache {
            return Err(MemoryError::OutOfMemory(kind));
        }

        let pool = self.pools.get(&kind).ok_or(MemoryError::DeviceNotFound(kind))?;
        if pool.can_allocate(required_elements) {
            return Ok(());
        }

        let mut candidates: Vec<(TensorBufferId, BufferPriority, Instant, usize)> = self.buffers
            .iter()
            .filter(|(_, b)| {
                location_to_kind(&b.location) == kind
                    && !b.pinned
                    && !self.upcoming_ids.contains(&b.id)
                    && !self.pinned_buffers.contains(&b.id)
            })
            .map(|(id, b)| (*id, b.metadata.priority, b.metadata.last_access, b.size_elements))
            .collect();

        candidates.sort_by(|a, b| {
            let prio_cmp = match (&a.1, &b.1) {
                (BufferPriority::Low, BufferPriority::Low) => std::cmp::Ordering::Equal,
                (BufferPriority::Low, _) => std::cmp::Ordering::Less,
                (_, BufferPriority::Low) => std::cmp::Ordering::Greater,
                (BufferPriority::Medium, BufferPriority::Medium) => std::cmp::Ordering::Equal,
                (BufferPriority::Medium, BufferPriority::High) => std::cmp::Ordering::Less,
                (BufferPriority::High, BufferPriority::Medium) => std::cmp::Ordering::Greater,
                (BufferPriority::High, BufferPriority::High) => std::cmp::Ordering::Equal,
            };
            if prio_cmp != std::cmp::Ordering::Equal {
                return prio_cmp;
            }
            let age_cmp = a.2.cmp(&b.2);
            if age_cmp != std::cmp::Ordering::Equal {
                return age_cmp;
            }
            b.3.cmp(&a.3)
        });

        let (vram_usage, ram_usage) = self.get_usage_ratios();

        for (id, _priority, _last_access, _size) in candidates {
            let metadata = self.buffers.get(&id).map(|b| b.metadata.clone()).unwrap();
            let current_tier = match kind {
                MemoryDeviceKind::DeviceVram(_) => MemoryTier::Vram,
                MemoryDeviceKind::HostRam => MemoryTier::Ram,
                _ => unreachable!(),
            };
            let target_tier = self.policy.decide_movement(
                &metadata,
                current_tier,
                vram_usage,
                ram_usage,
            );

            let target_kind = match target_tier {
                Some(MemoryTier::Ram) => MemoryDeviceKind::HostRam,
                Some(MemoryTier::Ssd) => MemoryDeviceKind::SsdCache,
                Some(MemoryTier::Vram) => continue,
                None => continue,
            };

            if tier_to_kind(target_tier.unwrap()) == kind {
                continue;
            }

            let target_pool = self.pools.get(&target_kind).unwrap();
            if !target_pool.can_allocate(metadata.size_elements) {
                continue;
            }

            match self.move_buffer(id, target_kind) {
                Ok(()) => {
                    if let Some(pool) = self.pools.get(&kind) {
                        if pool.can_allocate(required_elements) {
                            return Ok(());
                        }
                    }
                }
                Err(e) => {
                    eprintln!("[MemoryExecutor] Failed to evict buffer {:?} to {:?}: {:?}", id, target_kind, e);
                }
            }
        }

        Err(MemoryError::OutOfMemory(kind))
    }

    // --- Управление предстоящими буферами ---

    pub fn hint_upcoming_buffers(&mut self, ids: &[TensorBufferId]) {
        for &id in ids {
            self.upcoming_ids.insert(id);
        }
    }

    pub fn clear_upcoming(&mut self) {
        self.upcoming_ids.clear();
    }

    // --- Разрешение буфера ---

    pub fn resolve_buffer(
        &mut self,
        id: TensorBufferId,
        target: MemoryDeviceKind,
    ) -> Result<ResolvedBuffer<'_>, MemoryError> {
        self.move_buffer(id, target)?;
        let buffer = self.buffers.get_mut(&id).unwrap();
        buffer.use_count += 1;
        buffer.metadata.touch();
        Ok(ResolvedBuffer {
            buffer: buffer as *mut TensorBuffer,
            _owner: std::marker::PhantomData,
        })
    }

    pub fn release_buffer(&mut self, id: TensorBufferId) {
        if let Some(buffer) = self.buffers.get_mut(&id) {
            buffer.use_count = buffer.use_count.saturating_sub(1);
            if buffer.is_temp && buffer.use_count == 0 {
                let _ = self.deallocate_buffer(id);
            }
        }
    }

    // --- Удаление буфера ---

    pub fn deallocate_buffer(&mut self, id: TensorBufferId) -> Result<(), MemoryError> {
        self.pinned_buffers.remove(&id);
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
        if let Some(raw_id) = self.buffer_to_raw.remove(&id) {
            self.raw_registry.unregister(raw_id, &mut self.pools);
        }
        self.upcoming_ids.remove(&id);
        Ok(())
    }

    // --- Подсказки планировщика ---

    pub fn hint_access(&mut self, ids: &[TensorBufferId]) {
        for &id in ids {
            if let Some(buffer) = self.buffers.get_mut(&id) {
                if buffer.metadata.priority != BufferPriority::High {
                    buffer.metadata.priority = BufferPriority::High;
                }
                buffer.metadata.touch();
            }
        }
    }

    // --- Фоновый tick ---

    pub fn tick(&mut self) {
        self.tick_counter += 1;
        let (vram_usage, ram_usage) = self.get_usage_ratios();

        let buffer_ids: Vec<TensorBufferId> = self.buffers.keys().copied().collect();

        for id in buffer_ids {
            let (current_tier, _priority, _size, _pinned, metadata) = {
                let buffer = match self.buffers.get(&id) {
                    Some(b) => b,
                    None => continue,
                };
                if buffer.pinned || self.upcoming_ids.contains(&id) {
                    continue;
                }
                let tier = location_to_tier(&buffer.location);
                let metadata = buffer.metadata.clone();
                (tier, buffer.metadata.priority, buffer.size_elements, buffer.pinned, metadata)
            };

            if let Some(target_tier) = self.policy.decide_movement(
                &metadata,
                current_tier,
                vram_usage,
                ram_usage,
            ) {
                let target_kind = tier_to_kind(target_tier);
                if let Err(e) = self.move_buffer(id, target_kind) {
                    eprintln!("[MemoryExecutor] Failed to move buffer {:?}: {:?}", id, e);
                }
            }
        }

        for buffer in self.buffers.values_mut() {
            buffer.metadata.reset_period_counter();
        }
    }

    // --- Принудительное вытеснение ---

    pub fn force_evict(&mut self, target_free_mb: u64) -> Result<(), MemoryError> {
        let target_bytes = target_free_mb * 1024 * 1024;

        let mut candidates: Vec<(TensorBufferId, BufferPriority, Instant, usize)> = self.buffers
            .iter()
            .filter(|(_, b)| matches!(b.location, BufferLocation::DeviceVram(_)) && !b.pinned)
            .map(|(id, b)| (*id, b.metadata.priority, b.metadata.last_access, b.size_elements))
            .collect();

        candidates.sort_by_key(|(_, priority, last_access, _)| {
            let priority_order = match priority {
                BufferPriority::High => 0,
                BufferPriority::Medium => 1,
                BufferPriority::Low => 2,
            };
            (priority_order, *last_access)
        });

        let mut freed_bytes = 0u64;
        for (id, _priority, _last_access, size) in candidates {
            if freed_bytes >= target_bytes {
                break;
            }
            if let Some(buffer) = self.buffers.get(&id) {
                if buffer.pinned || self.upcoming_ids.contains(&id) {
                    continue;
                }
            }
            let target = if self.ram_usage_ratio() < 0.7 {
                MemoryDeviceKind::HostRam
            } else {
                MemoryDeviceKind::SsdCache
            };
            self.move_buffer(id, target)?;
            freed_bytes += (size * 4) as u64;
        }

        Ok(())
    }

    // ===================================================================
    // НОВЫЕ МЕТОДЫ ДЛЯ УПРАВЛЯЕМЫХ МАТРИЧНЫХ БУФЕРОВ
    // ===================================================================

    /// Зарегистрировать управляемый матричный буфер в реестре.
    pub fn register_matrix(
        &mut self,
        rows: usize,
        cols: usize,
        location: MemoryDeviceKind,
        priority: BufferPriority,
    ) -> MatrixBufferId {
        let id = MatrixBufferId(self.next_matrix_id.fetch_add(1, Ordering::SeqCst));
        let info = MatrixBufferInfo::new(id, rows, cols, location, priority);
        self.matrix_buffers.insert(id, info);
        id
    }

    /// Снять матричный буфер с учёта.
    pub fn unregister_matrix(&mut self, id: MatrixBufferId) {
        self.matrix_buffers.remove(&id);
    }

    /// Обновить время последнего доступа к матричному буферу.
    pub fn touch_matrix(&mut self, id: MatrixBufferId) {
        if let Some(info) = self.matrix_buffers.get_mut(&id) {
            info.touch();
        }
    }

    /// Получить метаданные матричного буфера.
    pub fn get_matrix_info(&self, id: MatrixBufferId) -> Option<&MatrixBufferInfo> {
        self.matrix_buffers.get(&id)
    }

    /// Получить мутабельные метаданные матричного буфера.
    pub fn get_matrix_info_mut(&mut self, id: MatrixBufferId) -> Option<&mut MatrixBufferInfo> {
        self.matrix_buffers.get_mut(&id)
    }

    /// Создать управляемый `MatrixBuffer` в указанной памяти.
    pub fn acquire_matrix(
        &mut self,
        rows: usize,
        cols: usize,
        location: MemoryDeviceKind,
        priority: BufferPriority,
    ) -> Result<MatrixBuffer, MemoryError> {
        let total = rows * cols;
        self.ensure_matrix_can_allocate(location, total)?;

        let matrix_id = self.register_matrix(rows, cols, location, priority);

        let memory = self
            .memory_arc
            .as_ref()
            .expect("MemoryExecutor::acquire_matrix called before set_self_arc")
            .clone();

        let buffer = match location {
            MemoryDeviceKind::HostRam => {
                self.reserve_memory(MemoryDeviceKind::HostRam, total)?;
                let data = vec![0.0f32; total];
                MatrixBuffer::from_cpu_parts(
                    rows,
                    cols,
                    data,
                    memory,
                    matrix_id,
                )
            }
            MemoryDeviceKind::DeviceVram(dev_id) => {
                let (gpu_buf, raw_id) = self.create_raw_gpu_buffer(dev_id, total)?;
                MatrixBuffer::from_gpu_parts(
                    rows,
                    cols,
                    gpu_buf,
                    raw_id,
                    dev_id,
                    memory,
                    matrix_id,
                )
            }
            MemoryDeviceKind::SsdCache => {
                let ssd = self
                    .ssd_cache
                    .as_ref()
                    .ok_or(MemoryError::DeviceNotFound(MemoryDeviceKind::SsdCache))?;
                let handle = ssd.allocate(total)?;
                let data = vec![0.0f32; total];
                ssd.write(&handle, &data)?;
                // Для SSD MatrixBuffer прямого конструктора пока нет.
                // Возвращаем ошибку, чтобы не использовать неполную реализацию.
                return Err(MemoryError::SsdError(
                    "Direct SSD MatrixBuffer creation is not yet supported".to_string(),
                ));
            }
        };

        Ok(buffer)
    }

    /// Выбрать оптимальное расположение для матрицы с учётом доступности.
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

    fn ensure_matrix_can_allocate(
        &self,
        location: MemoryDeviceKind,
        elements: usize,
    ) -> Result<(), MemoryError> {
        if self.can_allocate(location, elements) {
            Ok(())
        } else {
            Err(MemoryError::OutOfMemory(location))
        }
    }

    /// Вспомогательный метод: создаёт GPU-буфер и регистрирует его как raw.
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
                usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
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

    /// Перемещает данные `MatrixBuffer` между устройствами памяти.
    /// Обновляет внутреннее хранилище и метаданные в реестре.
    pub(crate) fn move_matrix_storage(
        &mut self,
        buffer: &mut MatrixBuffer,
        target: MemoryDeviceKind,
    ) -> Result<(), MemoryError> {
        let elements = buffer.size();
        let current = buffer.device_kind();

        if current == target {
            return Ok(());
        }

        match (&buffer.storage, target) {
            (BufferStorage::Cpu(data), MemoryDeviceKind::DeviceVram(dev_id)) => {
                let (gpu_buf, raw_id) = self.upload_vec_to_gpu(dev_id, data)?;
                buffer.set_storage(BufferStorage::Gpu {
                    buffer: gpu_buf,
                    raw_id,
                    device_id: dev_id,
                });
            }
            (BufferStorage::Gpu { buffer: src, raw_id, device_id }, MemoryDeviceKind::HostRam) => {
                let ctx = self
                    .gpu_contexts
                    .get(device_id)
                    .ok_or(MemoryError::DeviceNotFound(current))?;
                let data = self.download_gpu_to_vec(ctx.clone(), src)?;
                self.raw_registry.unregister(*raw_id, &mut self.pools);
                buffer.set_storage(BufferStorage::Cpu(data));
            }
            (BufferStorage::Cpu(data), MemoryDeviceKind::SsdCache) => {
                let ssd = self
                    .ssd_cache
                    .as_ref()
                    .ok_or(MemoryError::DeviceNotFound(MemoryDeviceKind::SsdCache))?;
                let handle = ssd.allocate(elements)?;
                ssd.write(&handle, data)?;
                buffer.set_storage(BufferStorage::SsdCache(handle));
            }
            (BufferStorage::SsdCache(handle), MemoryDeviceKind::HostRam) => {
                let ssd = self
                    .ssd_cache
                    .as_ref()
                    .ok_or(MemoryError::DeviceNotFound(current))?;
                let data = ssd.read(handle)?;
                ssd.deallocate(handle)?;
                buffer.set_storage(BufferStorage::Cpu(data));
            }
            (BufferStorage::Gpu { .. }, MemoryDeviceKind::SsdCache) => {
                self.move_matrix_storage(buffer, MemoryDeviceKind::HostRam)?;
                self.move_matrix_storage(buffer, MemoryDeviceKind::SsdCache)?;
            }
            (BufferStorage::SsdCache(_), MemoryDeviceKind::DeviceVram(dev_id)) => {
                self.move_matrix_storage(buffer, MemoryDeviceKind::HostRam)?;
                self.move_matrix_storage(buffer, MemoryDeviceKind::DeviceVram(dev_id))?;
            }
            _ => {
                return Err(MemoryError::DataNotInLocation(
                    TensorBufferId(0),
                    BufferLocation::HostRam,
                ));
            }
        }

        if let Some(id) = buffer.matrix_id() {
            if let Some(info) = self.get_matrix_info_mut(id) {
                info.set_location(target);
                info.touch();
            }
        }

        Ok(())
    }

    /// Выгружает CPU-вектор в GPU-буфер.
    fn upload_vec_to_gpu(
        &mut self,
        device_id: DeviceId,
        data: &[f32],
    ) -> Result<(Subbuffer<[f32]>, RawBufferId), MemoryError> {
        let ctx = self
            .gpu_contexts
            .get(&device_id)
            .ok_or(MemoryError::DeviceNotFound(MemoryDeviceKind::DeviceVram(device_id)))?;
        let elements = data.len();
        let bytes = (elements * std::mem::size_of::<f32>()) as u64;

        let (staging_buf, staging_raw) = self.temp_pool.acquire(
            MemoryDeviceKind::HostRam,
            elements,
            &self.gpu_contexts,
            &mut self.pools,
            &mut self.raw_registry,
        );

        {
            let mut write = staging_buf.write().map_err(|e| {
                MemoryError::SsdError(format!("Failed to write staging buffer: {}", e))
            })?;
            write.copy_from_slice(data);
        }

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
            bytes,
        )
        .map_err(|e| MemoryError::SsdError(format!("Failed to allocate GPU buffer: {}", e)))?;

        data_mover::copy_buffer_sync(ctx.clone(), staging_buf.clone(), gpu_buf.clone());

        self.temp_pool
            .release(MemoryDeviceKind::HostRam, staging_buf, staging_raw);

        let raw_id = self.raw_registry.register(
            device_id,
            bytes,
            MemoryTypeFilter::PREFER_DEVICE,
            &mut self.pools,
        );

        Ok((gpu_buf, raw_id))
    }

    /// Скачивает GPU-буфер в CPU-вектор.
    fn download_gpu_to_vec(
        &mut self,
        ctx: Arc<GpuContext>,
        src: &Subbuffer<[f32]>,
    ) -> Result<Vec<f32>, MemoryError> {
        let elements = src.len() as usize;
        let bytes = (elements * std::mem::size_of::<f32>()) as u64;

        let (staging_buf, staging_raw) = self.temp_pool.acquire(
            MemoryDeviceKind::HostRam,
            elements,
            &self.gpu_contexts,
            &mut self.pools,
            &mut self.raw_registry,
        );

        data_mover::copy_buffer_sync(ctx, src.clone(), staging_buf.clone());

        let data = {
            let guard = staging_buf.read().map_err(|e| {
                MemoryError::SsdError(format!("Failed to read staging buffer: {}", e))
            })?;
            guard.to_vec()
        };

        self.temp_pool
            .release(MemoryDeviceKind::HostRam, staging_buf, staging_raw);

        Ok(data)
    }

    /// Освобождает SSD-буфер по его дескриптору.
    pub fn deallocate_ssd(&mut self, handle: &super::ssd_cache::SsdHandle) {
        if let Some(ssd) = &self.ssd_cache {
            if let Err(e) = ssd.deallocate(handle) {
                eprintln!("[MemoryExecutor] Failed to deallocate SSD handle: {:?}", e);
            }
        }
    }
}

// Вспомогательные функции (приватные)

fn location_to_kind(loc: &BufferLocation) -> MemoryDeviceKind {
    match loc {
        BufferLocation::HostRam => MemoryDeviceKind::HostRam,
        BufferLocation::DeviceVram(id) => MemoryDeviceKind::DeviceVram(*id),
        BufferLocation::SsdCache(_) => MemoryDeviceKind::SsdCache,
    }
}

fn location_to_tier(loc: &BufferLocation) -> MemoryTier {
    match loc {
        BufferLocation::HostRam => MemoryTier::Ram,
        BufferLocation::DeviceVram(_) => MemoryTier::Vram,
        BufferLocation::SsdCache(_) => MemoryTier::Ssd,
    }
}

fn tier_to_kind(tier: MemoryTier) -> MemoryDeviceKind {
    match tier {
        MemoryTier::Ram => MemoryDeviceKind::HostRam,
        MemoryTier::Vram => MemoryDeviceKind::DeviceVram(DeviceId(0)),
        MemoryTier::Ssd => MemoryDeviceKind::SsdCache,
    }
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