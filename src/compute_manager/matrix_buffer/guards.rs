// src/compute_manager/matrix_buffer/guards.rs

use std::sync::{Arc, RwLock};

use crate::compute_manager::memory_executor::executor::MemoryExecutor;
use crate::compute_manager::memory_executor::matrix_entry::MatrixStorage;
use crate::compute_manager::memory_executor::matrix_id::MatrixBufferId;

/// RAII-гард для чтения данных, хранящий локальную копию.
///
/// При создании гарда данные из CPU-хранилища копируются в локальный вектор.
/// Это позволяет избежать удержания блокировки `RwLock<MemoryExecutor>` на время
/// использования, что исключает взаимные блокировки при одновременном чтении
/// и записи одного и того же `MemoryExecutor` в одном потоке.
pub struct MatrixReadGuard {
    data: Vec<f32>,
    rows: usize,
    cols: usize,
}

impl MatrixReadGuard {
    /// Создаёт гард, копируя данные из CPU-хранилища.
    ///
    /// Если запись отсутствует или хранилище не является CPU, паникует.
    pub(crate) fn new(memory: &Arc<RwLock<MemoryExecutor>>, id: MatrixBufferId) -> Self {
        let mem = memory.read().unwrap();
        let entry = mem
            .get_matrix_entry(id)
            .expect("MatrixReadGuard: entry not found in MemoryExecutor");
        match &entry.storage {
            MatrixStorage::Cpu(data) => Self {
                data: data.clone(),
                rows: entry.rows,
                cols: entry.cols,
            },
            _ => panic!("MatrixReadGuard: only CPU storage is supported for reading"),
        }
    }

    /// Возвращает ссылку на локальную копию данных.
    pub fn as_slice(&self) -> Option<&[f32]> {
        Some(&self.data)
    }

    /// Возвращает количество строк.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Возвращает количество столбцов.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Для GPU-буферов не поддерживается, всегда `None`.
    pub fn as_gpu_buffer(&self) -> Option<&vulkano::buffer::Subbuffer<[f32]>> {
        None
    }
}

/// RAII-гард для записи данных, хранящий локальную копию и записывающий
/// её обратно в `MemoryExecutor` при удалении.
pub struct MatrixWriteGuard {
    id: MatrixBufferId,
    memory: Arc<RwLock<MemoryExecutor>>,
    data: Vec<f32>,
    rows: usize,
    cols: usize,
    written: bool,
}

impl MatrixWriteGuard {
    /// Создаёт гард, копируя текущие данные из CPU-хранилища.
    ///
    /// Если запись отсутствует или хранилище не является CPU, паникует.
    pub(crate) fn new(memory: &Arc<RwLock<MemoryExecutor>>, id: MatrixBufferId) -> Self {
        let mem = memory.read().unwrap();
        let entry = mem
            .get_matrix_entry(id)
            .expect("MatrixWriteGuard: entry not found in MemoryExecutor");
        match &entry.storage {
            MatrixStorage::Cpu(data) => Self {
                id,
                memory: memory.clone(),
                data: data.clone(),
                rows: entry.rows,
                cols: entry.cols,
                written: false,
            },
            _ => panic!("MatrixWriteGuard: only CPU storage is supported for writing"),
        }
    }

    /// Возвращает мутабельную ссылку на локальную копию данных.
    pub fn as_slice_mut(&mut self) -> Option<&mut [f32]> {
        Some(&mut self.data)
    }

    /// Возвращает количество строк.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Возвращает количество столбцов.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Для GPU-буферов не поддерживается, всегда `None`.
    pub fn as_gpu_buffer(&self) -> Option<&vulkano::buffer::Subbuffer<[f32]>> {
        None
    }
}

impl Drop for MatrixWriteGuard {
    /// При удалении гарда записывает локальную копию обратно в `MemoryExecutor`.
    fn drop(&mut self) {
        if self.written {
            return;
        }
        let mut mem = self.memory.write().unwrap();
        if let Some(entry) = mem.get_matrix_entry_mut(self.id) {
            if let MatrixStorage::Cpu(data) = &mut entry.storage {
                *data = std::mem::take(&mut self.data);
                self.written = true;
            }
        }
    }
}