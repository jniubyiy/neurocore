// src/compute_manager/matrix_buffer/pool.rs

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::compute_manager::device_spec::DeviceId;
use crate::compute_manager::memory_executor::executor::MemoryExecutor;
use crate::compute_manager::memory_executor::policy::BufferPriority;
use crate::compute_manager::memory_executor::types::MemoryDeviceKind;
use crate::compute_manager::matrix_buffer::handle::MatrixBufferHandle;

/// Статистика использования пула временных матриц.
#[derive(Debug, Default, Clone)]
pub struct PoolStats {
    /// Количество созданных буферов (не из пула).
    pub created: usize,
    /// Количество переиспользованных буферов (взятых из пула).
    pub reused: usize,
    /// Количество возвращённых в пул буферов.
    pub released: usize,
    /// Количество удалённых буферов при очистке или превышении лимита.
    pub removed: usize,
}

/// Пул переиспользуемых [`MatrixBufferHandle`].
///
/// Свободные дескрипторы группируются по типу памяти, количеству строк и столбцов.
/// Пул не владеет физическими данными напрямую — они находятся в `MemoryExecutor`.
/// При получении из пула дескриптор помечается как непомещенный в пул,
/// при возврате — как удерживаемый пулом.
pub struct TempMatrixPool {
    /// Свободные дескрипторы, ключ – (тип памяти, rows, cols).
    free: HashMap<(MemoryDeviceKind, usize, usize), VecDeque<MatrixBufferHandle>>,
    /// Глобальный менеджер памяти.
    memory: Arc<Mutex<MemoryExecutor>>,
    /// Максимальное время простоя буфера, после которого он удаляется из пула.
    max_idle_age: Option<Duration>,
    /// Максимальное количество свободных буферов в пуле.
    max_pool_size: Option<usize>,
    /// Статистика использования пула.
    stats: PoolStats,
}

impl TempMatrixPool {
    /// Создаёт пустой пул, связанный с указанным менеджером памяти.
    pub fn new(memory: Arc<Mutex<MemoryExecutor>>) -> Self {
        Self {
            free: HashMap::new(),
            memory,
            max_idle_age: None,
            max_pool_size: None,
            stats: PoolStats::default(),
        }
    }

    /// Устанавливает максимальное время простоя буфера, после которого он будет удалён.
    pub fn with_max_idle_age(mut self, age: Duration) -> Self {
        self.max_idle_age = Some(age);
        self
    }

    /// Устанавливает максимальное количество свободных буферов в пуле.
    pub fn with_max_pool_size(mut self, size: usize) -> Self {
        self.max_pool_size = Some(size);
        self
    }

    /// Возвращает статистику использования пула.
    pub fn stats(&self) -> &PoolStats {
        &self.stats
    }

    /// Извлекает из пула или создаёт новый CPU-буфер заданного размера.
    pub fn acquire(&mut self, rows: usize, cols: usize) -> MatrixBufferHandle {
        self.acquire_kind(MemoryDeviceKind::HostRam, rows, cols)
    }

    /// Извлекает из пула или создаёт новый GPU-буфер заданного размера
    /// на указанном устройстве.
    pub fn acquire_gpu(
        &mut self,
        device_id: DeviceId,
        rows: usize,
        cols: usize,
    ) -> MatrixBufferHandle {
        self.acquire_kind(MemoryDeviceKind::DeviceVram(device_id), rows, cols)
    }

    /// Извлекает из пула или создаёт новый управляемый `MatrixBufferHandle`
    /// в указанной памяти с приоритетом `Medium`.
    pub fn acquire_matrix(
        &mut self,
        rows: usize,
        cols: usize,
        location: MemoryDeviceKind,
    ) -> MatrixBufferHandle {
        self.acquire_kind(location, rows, cols)
    }

    /// Внутренний метод получения буфера определённого типа памяти.
    fn acquire_kind(
        &mut self,
        kind: MemoryDeviceKind,
        rows: usize,
        cols: usize,
    ) -> MatrixBufferHandle {
        // Сначала очищаем устаревшие буферы.
        self.cleanup();

        let key = (kind, rows, cols);
        if let Some(queue) = self.free.get_mut(&key) {
            if let Some(handle) = queue.pop_front() {
                // Снимаем пометку pooled без изменения счётчика ссылок.
                {
                    let mut mem = self.memory.lock().unwrap();
                    if let Some(entry) = mem.get_matrix_entry_mut(handle.id()) {
                        entry.pooled = false;
                    }
                }
                self.stats.reused += 1;
                return handle;
            }
        }

        // Создаём новый буфер через MemoryExecutor.
        let handle = {
            let mut mem = self.memory.lock().unwrap();
            mem.acquire_matrix_handle(rows, cols, kind, BufferPriority::Medium)
                .expect("TempMatrixPool: failed to acquire MatrixBufferHandle")
        };
        self.stats.created += 1;
        handle
    }

    /// Возвращает буфер в пул для последующего переиспользования.
    ///
    /// Если на запись ссылается больше одного дескриптора (например, есть клоны),
    /// то буфер не помещается в пул, а просто сбрасывается.
    pub fn release(&mut self, handle: MatrixBufferHandle) {
        let id = handle.id();
        let (rows, cols) = (handle.rows(), handle.cols());
        let kind = handle.device_kind();

        // Проверяем, что на запись ссылается только этот handle.
        let should_pool = {
            let mut mem = self.memory.lock().unwrap();
            if let Some(entry) = mem.get_matrix_entry_mut(id) {
                if entry.ref_count > 1 {
                    // Есть другие владельцы — не кладём в пул, просто дропаем.
                    false
                } else {
                    // Помечаем как pooled, не меняя счётчик.
                    entry.pooled = true;
                    entry.touch();
                    true
                }
            } else {
                // Запись уже удалена.
                false
            }
        };
        // Блокировка снята.

        if !should_pool {
            // Запись используется другими владельцами или уже удалена.
            drop(handle);
            return;
        }

        // Кладём дескриптор в очередь. ref_count остаётся 1 (пул владеет).
        let key = (kind, rows, cols);
        self.free
            .entry(key)
            .or_insert_with(VecDeque::new)
            .push_back(handle);
        self.stats.released += 1;
        self.enforce_max_size();
    }

    /// Очищает все свободные буферы, возвращая зарезервированную память.
    pub fn clear(&mut self) {
        for list in self.free.values_mut() {
            while let Some(handle) = list.pop_front() {
                {
                    let mut mem = self.memory.lock().unwrap();
                    if let Some(entry) = mem.get_matrix_entry_mut(handle.id()) {
                        entry.pooled = false;
                    }
                }
                drop(handle);
                self.stats.removed += 1;
            }
        }
        self.free.clear();
    }

    /// Возвращает количество свободных буферов во всех категориях.
    pub fn free_count(&self) -> usize {
        self.free.values().map(|q| q.len()).sum()
    }

    /// Удаляет буферы, которые простаивали дольше `max_idle_age`,
    /// а также применяет ограничение по общему количеству.
    pub fn cleanup(&mut self) {
        if self.max_idle_age.is_none() && self.max_pool_size.is_none() {
            return;
        }

        let now = Instant::now();
        let mut empty_keys = Vec::new();

        if let Some(max_age) = self.max_idle_age {
            for (key, queue) in self.free.iter_mut() {
                let before = queue.len();
                let mut new_queue = VecDeque::new();
                while let Some(handle) = queue.pop_front() {
                    let last_access = {
                        let mem = self.memory.lock().unwrap();
                        mem.get_matrix_entry(handle.id())
                            .map(|e| e.last_access)
                            .unwrap_or(Instant::now() - max_age)
                    };
                    if now.duration_since(last_access) < max_age {
                        new_queue.push_back(handle);
                    } else {
                        {
                            let mut mem = self.memory.lock().unwrap();
                            if let Some(entry) = mem.get_matrix_entry_mut(handle.id()) {
                                entry.pooled = false;
                            }
                        }
                        drop(handle);
                        self.stats.removed += 1;
                    }
                }
                *queue = new_queue;
                if queue.is_empty() && before > 0 {
                    empty_keys.push(*key);
                }
            }
        }

        for key in empty_keys {
            self.free.remove(&key);
        }

        self.enforce_max_size();
    }

    /// Поддерживает лимит на количество свободных буферов.
    fn enforce_max_size(&mut self) {
        if let Some(limit) = self.max_pool_size {
            while self.free_count() > limit {
                let mut removed = false;
                for queue in self.free.values_mut() {
                    if let Some(handle) = queue.pop_back() {
                        {
                            let mut mem = self.memory.lock().unwrap();
                            if let Some(entry) = mem.get_matrix_entry_mut(handle.id()) {
                                entry.pooled = false;
                            }
                        }
                        drop(handle);
                        self.stats.removed += 1;
                        removed = true;
                        break;
                    }
                }
                if !removed {
                    break;
                }
            }
            self.free.retain(|_, q| !q.is_empty());
        }
    }
}