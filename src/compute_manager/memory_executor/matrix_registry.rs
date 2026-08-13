// src/compute_manager/memory_executor/matrix_registry.rs

use std::time::Instant;

use super::matrix_id::MatrixBufferId;
use super::policy::BufferPriority;
use super::types::MemoryDeviceKind;

/// Метаданные управляемого матричного буфера.
///
/// Хранят учётную информацию о `MatrixBuffer`, которая используется
/// `MemoryExecutor` для принятия решений о миграции, очистке и
/// отслеживании жизненного цикла.
#[derive(Debug, Clone)]
pub struct MatrixBufferInfo {
    /// Идентификатор буфера.
    pub id: MatrixBufferId,

    /// Количество строк матрицы.
    pub rows: usize,

    /// Количество столбцов матрицы.
    pub cols: usize,

    /// Общее количество элементов (`rows * cols`).
    pub total_elements: usize,

    /// Текущее физическое расположение данных.
    pub location: MemoryDeviceKind,

    /// Находится ли буфер в видеопамяти GPU.
    pub is_gpu: bool,

    /// Время последнего доступа к буферу.
    pub last_access: Instant,

    /// Приоритет удержания в быстрой памяти.
    pub priority: BufferPriority,

    /// Закреплён ли буфер (не подлежит автоматической миграции).
    pub pinned: bool,
}

impl MatrixBufferInfo {
    /// Создаёт новые метаданные для буфера.
    ///
    /// # Аргументы
    /// * `id` – уникальный идентификатор буфера.
    /// * `rows` – количество строк.
    /// * `cols` – количество столбцов.
    /// * `location` – начальное расположение данных.
    /// * `priority` – приоритет удержания.
    pub fn new(
        id: MatrixBufferId,
        rows: usize,
        cols: usize,
        location: MemoryDeviceKind,
        priority: BufferPriority,
    ) -> Self {
        Self {
            id,
            rows,
            cols,
            total_elements: rows * cols,
            location,
            is_gpu: matches!(location, MemoryDeviceKind::DeviceVram(_)),
            last_access: Instant::now(),
            priority,
            pinned: false,
        }
    }

    /// Обновляет время последнего доступа.
    #[inline]
    pub fn touch(&mut self) {
        self.last_access = Instant::now();
    }

    /// Обновляет расположение данных.
    #[inline]
    pub fn set_location(&mut self, location: MemoryDeviceKind) {
        self.location = location;
        self.is_gpu = matches!(location, MemoryDeviceKind::DeviceVram(_));
    }

    /// Закрепляет буфер, запрещая автоматическую миграцию.
    #[inline]
    pub fn pin(&mut self) {
        self.pinned = true;
    }

    /// Снимает закрепление с буфера.
    #[inline]
    pub fn unpin(&mut self) {
        self.pinned = false;
    }

    /// Возвращает `true`, если буфер закреплён.
    #[inline]
    pub fn is_pinned(&self) -> bool {
        self.pinned
    }
}