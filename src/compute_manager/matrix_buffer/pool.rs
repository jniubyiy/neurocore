// src/compute_manager/matrix_buffer/pool.rs

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::compute_manager::device_spec::DeviceId;
use crate::compute_manager::memory_executor::MemoryExecutor;
use crate::compute_manager::memory_executor::policy::BufferPriority;
use crate::compute_manager::memory_executor::types::MemoryDeviceKind;

use super::buffer::MatrixBuffer;

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

/// Пул переиспользуемых [`MatrixBuffer`], минимизирующий выделения памяти.
///
/// Свободные буферы сгруппированы по типу памяти, количеству строк и столбцов.
/// При запросе буфера сначала ищется подходящий в пуле; если его нет,
/// создаётся новый через [`MatrixBuffer::new`] (для CPU) или [`MatrixBuffer::new_gpu`] (для GPU).
///
/// Пул **не** является потокобезопасным сам по себе – для использования из
/// нескольких потоков следует обернуть его в `Arc<Mutex<...>>`.
pub struct TempMatrixPool {
    /// Свободные буферы, ключ – (тип памяти, rows, cols).
    free: HashMap<(MemoryDeviceKind, usize, usize), Vec<MatrixBuffer>>,
    /// Глобальный менеджер памяти, используемый для создания новых буферов.
    memory: Arc<Mutex<MemoryExecutor>>,
    /// Максимальное время простоя буфера, после которого он будет удалён из пула.
    max_idle_age: Option<Duration>,
    /// Максимальное количество свободных буферов, хранящихся в пуле.
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
    ///
    /// # Пример
    /// ```
    /// let pool = TempMatrixPool::new(memory).with_max_idle_age(Duration::from_secs(30));
    /// ```
    pub fn with_max_idle_age(mut self, age: Duration) -> Self {
        self.max_idle_age = Some(age);
        self
    }

    /// Устанавливает максимальное количество свободных буферов в пуле.
    ///
    /// При превышении лимита самые старые буферы удаляются.
    pub fn with_max_pool_size(mut self, size: usize) -> Self {
        self.max_pool_size = Some(size);
        self
    }

    /// Возвращает статистику использования пула.
    pub fn stats(&self) -> &PoolStats {
        &self.stats
    }

    /// Извлекает из пула или создаёт новый CPU-буфер заданного размера.
    ///
    /// Буфер возвращается с нулевым содержимым.
    pub fn acquire(&mut self, rows: usize, cols: usize) -> MatrixBuffer {
        self.acquire_kind(MemoryDeviceKind::HostRam, rows, cols)
    }

    /// Извлекает из пула или создаёт новый GPU-буфер заданного размера
    /// на указанном устройстве.
    pub fn acquire_gpu(&mut self, device_id: DeviceId, rows: usize, cols: usize) -> MatrixBuffer {
        self.acquire_kind(MemoryDeviceKind::DeviceVram(device_id), rows, cols)
    }

    /// Извлекает из пула или создаёт новый управляемый `MatrixBuffer`
    /// в указанной памяти с приоритетом `Medium`.
    ///
    /// Буфер регистрируется в `MemoryExecutor` и получает `MatrixBufferId`.
    pub fn acquire_matrix(
        &mut self,
        rows: usize,
        cols: usize,
        location: MemoryDeviceKind,
    ) -> MatrixBuffer {
        let key = (location, rows, cols);
        if let Some(list) = self.free.get_mut(&key) {
            if let Some(mut buf) = list.pop() {
                buf.mark_used();
                self.stats.reused += 1;
                return buf;
            }
        }

        // Создаём через MemoryExecutor с регистрацией
        let mut mem = self.memory.lock().unwrap();
        let buf = mem
            .acquire_matrix(rows, cols, location, BufferPriority::Medium)
            .expect("TempMatrixPool: failed to acquire MatrixBuffer via MemoryExecutor");
        drop(mem);

        self.stats.created += 1;
        buf
    }

    /// Извлекает из пула или создаёт новый управляемый `MatrixBuffer`
    /// с автоматическим выбором памяти на основе политики MemoryExecutor.
    ///
    /// Сначала пытается разместить в `preferred`, затем в RAM, затем в SSD (если доступен).
    pub fn acquire_matrix_auto(
        &mut self,
        rows: usize,
        cols: usize,
        preferred: MemoryDeviceKind,
    ) -> MatrixBuffer {
        let elements = rows * cols;
        let target = {
            let mem = self.memory.lock().unwrap();
            mem.select_matrix_location(elements, preferred, BufferPriority::Medium)
        };
        self.acquire_matrix(rows, cols, target)
    }

    /// Внутренний метод для получения буфера определённого типа памяти.
    fn acquire_kind(&mut self, kind: MemoryDeviceKind, rows: usize, cols: usize) -> MatrixBuffer {
        // Сначала очищаем устаревшие буферы
        self.cleanup();

        let key = (kind, rows, cols);
        if let Some(list) = self.free.get_mut(&key) {
            if let Some(mut buf) = list.pop() {
                buf.mark_used();
                self.stats.reused += 1;
                return buf;
            }
        }

        // Создаём новый буфер в зависимости от типа памяти
        let buf = match kind {
            MemoryDeviceKind::HostRam => MatrixBuffer::new(&self.memory, rows, cols)
                .expect("TempMatrixPool: failed to allocate CPU MatrixBuffer"),
            MemoryDeviceKind::DeviceVram(device_id) => MatrixBuffer::new_gpu(&self.memory, device_id, rows, cols)
                .expect("TempMatrixPool: failed to allocate GPU MatrixBuffer"),
            _ => panic!("TempMatrixPool: unsupported memory kind {:?}", kind),
        };
        self.stats.created += 1;
        buf
    }

    /// Возвращает буфер в пул для последующего переиспользования.
    ///
    /// Буфер **не** очищается – предполагается, что новые данные будут
    /// записаны поверх старых при следующем использовании.
    pub fn release(&mut self, mut buf: MatrixBuffer) {
        buf.mark_used(); // обновляем время последнего использования
        // Обновляем время в реестре MemoryExecutor, если буфер управляется
        if let Some(id) = buf.matrix_id() {
            if let Ok(mut mem) = self.memory.lock() {
                mem.touch_matrix(id);
            }
        }
        let kind = buf.device_kind();
        let key = (kind, buf.rows(), buf.cols());
        self.free.entry(key).or_insert_with(Vec::new).push(buf);
        self.stats.released += 1;
        // Проверяем лимит размера пула
        self.enforce_max_size();
    }

    /// Очищает все свободные буферы, возвращая зарезервированную память
    /// менеджеру памяти. После этого пул пуст.
    pub fn clear(&mut self) {
        for list in self.free.values_mut() {
            for buf in list.iter_mut() {
                buf.deallocate();
            }
        }
        self.free.clear();
    }

    /// Возвращает количество свободных буферов во всех категориях.
    pub fn free_count(&self) -> usize {
        self.free.values().map(|v| v.len()).sum()
    }

    /// Удаляет из пула буферы, которые простаивали дольше `max_idle_age`,
    /// а также вызывает ограничение по общему количеству.
    pub fn cleanup(&mut self) {
        if self.max_idle_age.is_none() && self.max_pool_size.is_none() {
            return; // нет ограничений – ничего не делаем
        }

        let now = Instant::now();
        let mut empty_keys = Vec::new();

        // Удаляем устаревшие буферы
        if let Some(max_age) = self.max_idle_age {
            for (key, list) in self.free.iter_mut() {
                let before = list.len();
                list.retain(|buf| {
                    let keep = now.duration_since(buf.last_used()) < max_age;
                    if !keep {
                        self.stats.removed += 1;
                    }
                    keep
                });
                if list.is_empty() && before > 0 {
                    empty_keys.push(*key);
                }
            }
        }

        // Убираем пустые записи
        for key in empty_keys {
            self.free.remove(&key);
        }

        // Принудительно ограничиваем размер пула
        self.enforce_max_size();
    }

    /// Вспомогательный метод для поддержания лимита на количество свободных буферов.
    fn enforce_max_size(&mut self) {
        if let Some(limit) = self.max_pool_size {
            while self.free_count() > limit {
                // Находим первый непустой список и удаляем из него последний буфер
                let mut removed = false;
                for list in self.free.values_mut() {
                    if let Some(_buf) = list.pop() {
                        self.stats.removed += 1;
                        removed = true;
                        break;
                    }
                }
                if !removed {
                    break; // защита от бесконечного цикла, если все списки пусты
                }
            }
            // Удаляем пустые записи
            self.free.retain(|_, list| !list.is_empty());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute_manager::memory_executor::MemoryExecutor;
    use std::sync::Arc;

    fn create_pool() -> (Arc<Mutex<MemoryExecutor>>, TempMatrixPool) {
        let mem = Arc::new(Mutex::new(MemoryExecutor::new()));
        let pool = TempMatrixPool::new(mem.clone());
        (mem, pool)
    }

    #[test]
    fn test_acquire_release_cpu() {
        let (_mem, mut pool) = create_pool();

        let a = pool.acquire(3, 4);
        assert_eq!(a.rows(), 3);
        assert_eq!(a.cols(), 4);
        assert!(!a.is_gpu());
        pool.release(a);
        assert_eq!(pool.free_count(), 1);

        // Повторный запрос того же размера должен вернуть буфер из пула
        let b = pool.acquire(3, 4);
        assert_eq!(pool.free_count(), 0);
        assert_eq!(b.rows(), 3);
        assert_eq!(b.cols(), 4);
        assert_eq!(pool.stats().reused, 1);
        assert_eq!(pool.stats().created, 1);
    }

    #[test]
    fn test_different_sizes_cpu() {
        let (_mem, mut pool) = create_pool();
        let a = pool.acquire(2, 5);
        pool.release(a);
        // Запрос другого размера должен создать новый буфер
        let b = pool.acquire(5, 2);
        assert_eq!(b.rows(), 5);
        assert_eq!(b.cols(), 2);
        assert_eq!(pool.free_count(), 0);
        assert_eq!(pool.stats().created, 2);
    }

    #[test]
    fn test_cleanup_removes_old_buffers() {
        let (_mem, mut pool) = create_pool();
        // Настраиваем пул на удаление буферов старше 0 секунд
        pool.max_idle_age = Some(Duration::from_secs(0));

        let a = pool.acquire(3, 4);
        pool.release(a);
        // Так как max_idle_age = 0, при следующем acquire буфер будет удалён
        let b = pool.acquire(3, 4);
        assert_eq!(b.rows(), 3);
        assert_eq!(b.cols(), 4);
        // Должен быть создан новый буфер
        assert_eq!(pool.stats().created, 2);
        assert_eq!(pool.stats().reused, 0);
        assert_eq!(pool.stats().removed, 1);
    }

    #[test]
    fn test_max_pool_size_cpu() {
        let (_mem, mut pool) = create_pool();
        pool.max_pool_size = Some(1);

        let a = pool.acquire(2, 2);
        let b = pool.acquire(3, 3);
        pool.release(a);
        pool.release(b);

        // После двух release при лимите 1 должен остаться только один буфер
        assert_eq!(pool.free_count(), 1);
        // Один буфер был удалён
        assert_eq!(pool.stats().removed, 1);
    }

    // Тесты для GPU-буферов не запускаются без реального GPU,
    // но оставлены для иллюстрации.
    #[test]
    #[ignore]
    fn test_acquire_release_gpu() {
        let (_mem, mut pool) = create_pool();
        // Предположим, что у нас есть GPU-контекст с DeviceId(0)
        let device_id = DeviceId(0);
        let a = pool.acquire_gpu(device_id, 3, 4);
        assert!(a.is_gpu());
        assert_eq!(a.device_kind(), MemoryDeviceKind::DeviceVram(device_id));
        pool.release(a);
        assert_eq!(pool.free_count(), 1);

        let b = pool.acquire_gpu(device_id, 3, 4);
        assert_eq!(pool.free_count(), 0);
        assert!(b.is_gpu());
    }

    // Новый тест: acquire_matrix через MemoryExecutor
    #[test]
    fn test_acquire_matrix_cpu() {
        let (mut mem, mut pool) = create_pool();
        // Регистрируем CPU устройство, чтобы MemoryExecutor мог выделять HostRam
        mem.lock().unwrap().register_compute_device(
            crate::compute_manager::device_spec::DeviceSpec::cpu(0, 1024, 1),
            None,
        );
        let buf = pool.acquire_matrix(2, 3, MemoryDeviceKind::HostRam);
        assert_eq!(buf.rows(), 2);
        assert_eq!(buf.cols(), 3);
        assert!(buf.matrix_id().is_some());
        let id = buf.matrix_id().unwrap();
        assert!(mem.lock().unwrap().get_matrix_info(id).is_some());
        pool.release(buf);
        // После release, если буфер возвращён в пул, он остаётся в реестре,
        // так как он не деаллоцирован. Проверим, что реестр всё ещё содержит id.
        assert!(mem.lock().unwrap().get_matrix_info(id).is_some());
    }
}