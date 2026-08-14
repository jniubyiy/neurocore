// src/compute_manager/matrix_buffer/weak_handle.rs

use std::sync::{Arc, Mutex};

use crate::compute_manager::memory_executor::executor::MemoryExecutor;
use crate::compute_manager::memory_executor::matrix_id::MatrixBufferId;

use super::handle::MatrixBufferHandle;

/// Слабая ссылка на управляемый матричный буфер.
///
/// В отличие от [`MatrixBufferHandle`], слабая ссылка **не увеличивает**
/// счётчик активных дескрипторов в `MemoryExecutor`. Это позволяет хранить
/// необязательные ссылки на буферы без предотвращения их освобождения.
///
/// Преобразовать слабую ссылку в сильную можно с помощью [`upgrade`],
/// который вернёт `Some(MatrixBufferHandle)` только если запись всё ещё
/// существует и имеет ненулевой счётчик активных дескрипторов.
pub struct WeakMatrixBufferHandle {
    pub(crate) id: MatrixBufferId,
    pub(crate) memory: Arc<Mutex<MemoryExecutor>>,
}

impl WeakMatrixBufferHandle {
    /// Создаёт слабую ссылку из сильного дескриптора.
    ///
    /// Счётчик ссылок в `MemoryExecutor` при этом не изменяется.
    pub fn from_handle(handle: &MatrixBufferHandle) -> Self {
        Self {
            id: handle.id(),
            memory: handle.memory().clone(),
        }
    }

    /// Возвращает уникальный идентификатор буфера.
    #[inline]
    pub fn id(&self) -> MatrixBufferId {
        self.id
    }

    /// Пытается преобразовать слабую ссылку в сильный дескриптор.
    ///
    /// Возвращает `Some(MatrixBufferHandle)`, если:
    /// - запись с данным идентификатором существует в `MemoryExecutor`;
    /// - счётчик активных дескрипторов больше нуля.
    ///
    /// При успешном преобразовании счётчик ссылок увеличивается на 1.
    pub fn upgrade(&self) -> Option<MatrixBufferHandle> {
        let mut mem = self.memory.lock().unwrap();
        let entry = mem.get_matrix_entry(self.id)?;

        if entry.ref_count == 0 {
            return None;
        }

        mem.increment_ref_count(self.id);
        Some(MatrixBufferHandle::new(self.id, self.memory.clone()))
    }

    /// Возвращает `true`, если запись существует и имеет ненулевой счётчик.
    pub fn is_alive(&self) -> bool {
        let mem = self.memory.lock().unwrap();
        mem.get_matrix_entry(self.id)
            .map(|entry| entry.ref_count > 0)
            .unwrap_or(false)
    }
}

impl Clone for WeakMatrixBufferHandle {
    /// Клонирование слабой ссылки не меняет счётчик активных дескрипторов.
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            memory: self.memory.clone(),
        }
    }
}

impl std::fmt::Debug for WeakMatrixBufferHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WeakMatrixBufferHandle")
            .field("id", &self.id)
            .finish()
    }
}