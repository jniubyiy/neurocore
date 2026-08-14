// src/compute_manager/memory_executor/matrix_entry.rs

use std::time::Instant;
use vulkano::buffer::Subbuffer;

use crate::compute_manager::device_spec::DeviceId;
use crate::compute_manager::memory_executor::policy::BufferPriority;
use crate::compute_manager::memory_executor::raw_buffer::RawBufferId;
use crate::compute_manager::memory_executor::ssd_cache::SsdHandle;
use crate::compute_manager::memory_executor::types::MemoryDeviceKind;

/// Физическое хранилище данных матрицы.
///
/// Может находиться в оперативной памяти (CPU), видеопамяти GPU или на SSD.
/// Все варианты владеют своими ресурсами и освобождаются при удалении записи
/// из `MemoryExecutor`.
#[derive(Debug, Clone)]
pub enum MatrixStorage {
    /// Данные в оперативной памяти (column‑major порядок).
    Cpu(Vec<f32>),

    /// Данные в видеопамяти GPU.
    Gpu {
        buffer: Subbuffer<[f32]>,
        raw_id: RawBufferId,
        device_id: DeviceId,
    },

    /// Данные выгружены на SSD.
    Ssd(SsdHandle),
}

impl MatrixStorage {
    /// Возвращает `true`, если данные находятся на GPU.
    pub fn is_gpu(&self) -> bool {
        matches!(self, MatrixStorage::Gpu { .. })
    }

    /// Возвращает `true`, если данные находятся в оперативной памяти.
    pub fn is_cpu(&self) -> bool {
        matches!(self, MatrixStorage::Cpu(_))
    }

    /// Возвращает `true`, если данные находятся на SSD.
    pub fn is_ssd(&self) -> bool {
        matches!(self, MatrixStorage::Ssd(_))
    }
}

/// Полная запись о матричном буфере в реестре `MemoryExecutor`.
///
/// Содержит физические данные, размеры, счётчик активных дескрипторов
/// и метаданные для управления памятью.
#[derive(Debug)]
pub struct MatrixEntry {
    /// Количество строк.
    pub rows: usize,

    /// Количество столбцов.
    pub cols: usize,

    /// Физическое хранилище данных.
    pub storage: MatrixStorage,

    /// Количество активных `MatrixBufferHandle`, ссылающихся на эту запись.
    /// Когда достигает нуля и `pooled == false`, запись удаляется из реестра.
    pub ref_count: usize,

    /// Флаг, указывающий, что запись удерживается пулом временных матриц.
    /// Такие записи не удаляются при нулевом счётчике ссылок, пока не будут
    /// явно изъяты из пула или очищены.
    pub pooled: bool,

    /// Время последнего доступа (для политики вытеснения).
    pub last_access: Instant,

    /// Приоритет удержания в быстрой памяти.
    pub priority: BufferPriority,

    /// Закреплена ли запись (не подлежит автоматическому перемещению).
    pub pinned: bool,
}

impl MatrixEntry {
    /// Создаёт новую запись с указанными размерами, хранилищем и приоритетом.
    ///
    /// Счётчик ссылок инициализируется единицей, так как при создании
    /// возвращается один дескриптор.
    pub fn new(
        rows: usize,
        cols: usize,
        storage: MatrixStorage,
        priority: BufferPriority,
    ) -> Self {
        Self {
            rows,
            cols,
            storage,
            ref_count: 1,
            pooled: false,
            last_access: Instant::now(),
            priority,
            pinned: false,
        }
    }

    /// Общее количество элементов матрицы.
    pub fn size(&self) -> usize {
        self.rows * self.cols
    }

    /// Обновляет время последнего доступа.
    pub fn touch(&mut self) {
        self.last_access = Instant::now();
    }

    /// Возвращает `true`, если данные находятся на GPU.
    pub fn is_gpu(&self) -> bool {
        self.storage.is_gpu()
    }

    /// Возвращает `true`, если данные находятся в оперативной памяти.
    pub fn is_cpu(&self) -> bool {
        self.storage.is_cpu()
    }

    /// Возвращает `true`, если данные находятся на SSD.
    pub fn is_ssd(&self) -> bool {
        self.storage.is_ssd()
    }

    /// Возвращает тип устройства памяти, на котором находятся данные.
    pub fn device_kind(&self) -> MemoryDeviceKind {
        match &self.storage {
            MatrixStorage::Cpu(_) => MemoryDeviceKind::HostRam,
            MatrixStorage::Gpu { device_id, .. } => MemoryDeviceKind::DeviceVram(*device_id),
            MatrixStorage::Ssd(_) => MemoryDeviceKind::SsdCache,
        }
    }
}