// src/compute_manager/matrix_buffer/pool.rs

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::compute_manager::memory_executor::MemoryExecutor;

use super::buffer::MatrixBuffer;

/// Пул переиспользуемых [`MatrixBuffer`], минимизирующий выделения памяти.
///
/// Свободные буферы сгруппированы по точным размерам `(rows, cols)`.
/// При запросе буфера сначала ищется подходящий в пуле; если его нет,
/// создаётся новый через [`MatrixBuffer::new`].
///
/// Пул **не** является потокобезопасным сам по себе – для использования из
/// нескольких потоков следует обернуть его в `Arc<Mutex<...>>`.
pub struct TempMatrixPool {
    /// Свободные буферы, ключ – размеры (rows, cols).
    free: HashMap<(usize, usize), Vec<MatrixBuffer>>,
    /// Глобальный менеджер памяти, используемый для создания новых буферов.
    memory: Arc<Mutex<MemoryExecutor>>,
}

impl TempMatrixPool {
    /// Создаёт пустой пул, связанный с указанным менеджером памяти.
    pub fn new(memory: Arc<Mutex<MemoryExecutor>>) -> Self {
        Self {
            free: HashMap::new(),
            memory,
        }
    }

    /// Извлекает из пула или создаёт новый буфер заданного размера.
    ///
    /// Буфер возвращается с нулевым содержимым.
    pub fn acquire(&mut self, rows: usize, cols: usize) -> MatrixBuffer {
        let key = (rows, cols);
        if let Some(list) = self.free.get_mut(&key) {
            if let Some(buf) = list.pop() {
                return buf;
            }
        }
        // Создаём новый, паника при нехватке памяти (ошибка не обрабатывается)
        MatrixBuffer::new(&self.memory, rows, cols)
            .expect("TempMatrixPool: failed to allocate MatrixBuffer")
    }

    /// Возвращает буфер в пул для последующего переиспользования.
    ///
    /// Буфер **не** очищается – предполагается, что новые данные будут
    /// записаны поверх старых при следующем использовании.
    pub fn release(&mut self, buf: MatrixBuffer) {
        let key = (buf.rows(), buf.cols());
        self.free.entry(key).or_insert_with(Vec::new).push(buf);
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute_manager::memory_executor::MemoryExecutor;
    use std::sync::Arc;

    #[test]
    fn test_acquire_release() {
        let mem = Arc::new(Mutex::new(MemoryExecutor::new()));
        let mut pool = TempMatrixPool::new(mem.clone());

        let a = pool.acquire(3, 4);
        assert_eq!(a.rows(), 3);
        assert_eq!(a.cols(), 4);
        pool.release(a);
        assert_eq!(pool.free_count(), 1);

        // Повторный запрос того же размера должен вернуть буфер из пула
        let b = pool.acquire(3, 4);
        assert_eq!(pool.free_count(), 0);
        // Признак переиспользования – размеры совпадают
        assert_eq!(b.rows(), 3);
        assert_eq!(b.cols(), 4);
    }

    #[test]
    fn test_different_sizes() {
        let mem = Arc::new(Mutex::new(MemoryExecutor::new()));
        let mut pool = TempMatrixPool::new(mem);
        let a = pool.acquire(2, 5);
        pool.release(a);
        // Запрос другого размера должен создать новый буфер
        let b = pool.acquire(5, 2);
        assert_eq!(b.rows(), 5);
        assert_eq!(b.cols(), 2);
        assert_eq!(pool.free_count(), 0);
    }
}