// src/compute_manager/matrix_buffer/handle.rs

use std::sync::{Arc, Mutex};

use crate::compute_manager::memory_executor::executor::MemoryExecutor;
use crate::compute_manager::memory_executor::matrix_entry::MatrixStorage;
use crate::compute_manager::memory_executor::matrix_id::MatrixBufferId;
use crate::compute_manager::memory_executor::types::MemoryDeviceKind;

use super::guards::{MatrixReadGuard, MatrixWriteGuard};
use super::weak_handle::WeakMatrixBufferHandle;

/// Лёгкий дескриптор управляемого матричного буфера.
///
/// Не владеет данными, а ссылается на запись в [`MemoryExecutor`].
/// Клонирование дескриптора дёшево и увеличивает счётчик ссылок,
/// позволяя безопасно разделять один буфер между несколькими потребителями.
pub struct MatrixBufferHandle {
    id: MatrixBufferId,
    memory: Arc<Mutex<MemoryExecutor>>,
}

impl MatrixBufferHandle {
    /// Создаёт новый дескриптор для указанной записи.
    ///
    /// Этот конструктор вызывается только из `MemoryExecutor` после
    /// успешного создания записи. Счётчик ссылок записи должен быть
    /// уже увеличен до 1.
    pub(crate) fn new(id: MatrixBufferId, memory: Arc<Mutex<MemoryExecutor>>) -> Self {
        Self { id, memory }
    }

    /// Возвращает уникальный идентификатор буфера.
    #[inline]
    pub fn id(&self) -> MatrixBufferId {
        self.id
    }

    /// Возвращает количество строк матрицы.
    ///
    /// # Паника
    /// Паникует, если запись была удалена из `MemoryExecutor`.
    pub fn rows(&self) -> usize {
        let mem = self.memory.lock().unwrap();
        mem.get_matrix_entry(self.id)
            .expect("MatrixBufferHandle: entry not found in MemoryExecutor")
            .rows
    }

    /// Возвращает количество столбцов матрицы.
    ///
    /// # Паника
    /// Паникует, если запись была удалена из `MemoryExecutor`.
    pub fn cols(&self) -> usize {
        let mem = self.memory.lock().unwrap();
        mem.get_matrix_entry(self.id)
            .expect("MatrixBufferHandle: entry not found in MemoryExecutor")
            .cols
    }

    /// Возвращает `true`, если данные находятся в видеопамяти GPU.
    pub fn is_gpu(&self) -> bool {
        let mem = self.memory.lock().unwrap();
        mem.get_matrix_entry(self.id)
            .expect("MatrixBufferHandle: entry not found in MemoryExecutor")
            .is_gpu()
    }

    /// Возвращает тип устройства памяти, на котором находятся данные.
    pub fn device_kind(&self) -> MemoryDeviceKind {
        let mem = self.memory.lock().unwrap();
        mem.get_matrix_entry(self.id)
            .expect("MatrixBufferHandle: entry not found in MemoryExecutor")
            .device_kind()
    }

    /// Запрашивает доступ на чтение к данным.
    ///
    /// Возвращает RAII-гард. Для CPU-буферов гард содержит локальную копию
    /// данных, поэтому блокировка `MemoryExecutor` не удерживается.
    pub fn read(&self) -> MatrixReadGuard {
        MatrixReadGuard::new(&self.memory, self.id)
    }

    /// Запрашивает доступ на запись к данным.
    ///
    /// Возвращает RAII-гард. Для CPU-буферов гард содержит локальную копию
    /// данных, которая записывается обратно в `MemoryExecutor` при удалении.
    pub fn write(&self) -> MatrixWriteGuard {
        MatrixWriteGuard::new(&self.memory, self.id)
    }

    /// Возвращает клонированную ссылку на `MemoryExecutor`.
    ///
    /// Используется внутри системы, в частности для создания слабых ссылок.
    #[inline]
    pub(crate) fn memory(&self) -> Arc<Mutex<MemoryExecutor>> {
        self.memory.clone()
    }

    /// Создаёт слабую ссылку на этот буфер.
    ///
    /// Слабая ссылка не увеличивает счётчик активных дескрипторов,
    /// поэтому запись может быть освобождена, даже если слабая ссылка существует.
    pub fn downgrade(&self) -> WeakMatrixBufferHandle {
        WeakMatrixBufferHandle {
            id: self.id,
            memory: self.memory.clone(),
        }
    }

    /// Копирует часть данных из буфера в новый вектор.
    ///
    /// Для CPU-хранилища выполняется прямое копирование диапазона
    /// без создания полной временной копии. Для GPU/SSD используется
    /// fallback на полное чтение.
    ///
    /// # Паника
    /// Паникует, если диапазон выходит за пределы данных.
    pub fn read_range(&self, start: usize, len: usize) -> Vec<f32> {
        let mem = self.memory.lock().unwrap();
        let entry = mem
            .get_matrix_entry(self.id)
            .expect("MatrixBufferHandle: entry not found in MemoryExecutor");

        match &entry.storage {
            MatrixStorage::Cpu(data) => {
                assert!(
                    start + len <= data.len(),
                    "MatrixBufferHandle::read_range: out of bounds (start={}, len={}, total={})",
                    start,
                    len,
                    data.len()
                );
                data[start..start + len].to_vec()
            }
            _ => {
                // Освобождаем блокировку перед вызовом read(),
                // так как read() снова блокирует MemoryExecutor.
                drop(mem);
                let guard = self.read();
                let slice = guard
                    .as_slice()
                    .expect("MatrixBufferHandle::read_range: expected CPU-compatible buffer");
                assert!(
                    start + len <= slice.len(),
                    "MatrixBufferHandle::read_range: out of bounds in fallback"
                );
                slice[start..start + len].to_vec()
            }
        }
    }

    /// Записывает данные в указанный диапазон буфера.
    ///
    /// Для CPU-хранилища выполняется прямая запись в диапазон
    /// без промежуточного копирования всего буфера. Для GPU/SSD
    /// используется fallback на полное чтение-модификацию-запись.
    ///
    /// # Паника
    /// Паникует, если диапазон выходит за пределы данных или
    /// длина `src` не соответствует диапазону.
    pub fn write_range(&self, start: usize, src: &[f32]) {
        let mut mem = self.memory.lock().unwrap();
        if let Some(entry) = mem.get_matrix_entry_mut(self.id) {
            if let MatrixStorage::Cpu(data) = &mut entry.storage {
                assert!(
                    start + src.len() <= data.len(),
                    "MatrixBufferHandle::write_range: out of bounds (start={}, len={}, total={})",
                    start,
                    src.len(),
                    data.len()
                );
                data[start..start + src.len()].copy_from_slice(src);
                return;
            }
        }

        // Освобождаем блокировку перед вызовом write(),
        // так как write() снова блокирует MemoryExecutor.
        drop(mem);

        let mut guard = self.write();
        let slice = guard
            .as_slice_mut()
            .expect("MatrixBufferHandle::write_range: expected CPU-compatible buffer");
        assert!(
            start + src.len() <= slice.len(),
            "MatrixBufferHandle::write_range: out of bounds in fallback"
        );
        slice[start..start + src.len()].copy_from_slice(src);
    }
}

impl Clone for MatrixBufferHandle {
    /// Клонирование дескриптора увеличивает счётчик ссылок в `MemoryExecutor`.
    ///
    /// Физические данные не копируются.
    fn clone(&self) -> Self {
        let mut mem = self.memory.lock().unwrap();
        mem.increment_ref_count(self.id);
        Self {
            id: self.id,
            memory: self.memory.clone(),
        }
    }
}

impl Drop for MatrixBufferHandle {
    /// При удалении дескриптора уменьшает счётчик ссылок. Если счётчик
    /// становится нулевым и запись не удерживается пулом, `MemoryExecutor`
    /// освобождает физическую память и удаляет запись.
    fn drop(&mut self) {
        if let Ok(mut mem) = self.memory.lock() {
            mem.release_matrix_handle(self.id);
        }
    }
}