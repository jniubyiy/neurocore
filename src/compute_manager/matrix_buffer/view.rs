// src/compute_manager/matrix_buffer/view.rs

use crate::compute_manager::matrix_buffer::handle::MatrixBufferHandle;
use crate::compute_manager::memory_executor::types::MemoryDeviceKind;

/// Лёгкое представление непрерывного диапазона элементов внутри родительского
/// управляемого матричного буфера [`MatrixBufferHandle`].
///
/// Не владеет данными и не создаёт отдельного буфера. Все операции чтения/записи
/// делегируются родительскому буферу с учётом смещения `offset_elements`.
///
/// Это позволяет эффективно работать с частями большого буфера (например,
/// с параметрами отдельного слоя, размещёнными в общем буфере сегмента)
/// без копирования целого буфера.
///
/// Для GPU-буферов можно получить родительский дескриптор и смещение, чтобы
/// внешний код мог создать `Subbuffer` нужного диапазона (например, через
/// `GpuCompute`).
#[derive(Clone)]
pub struct MatrixBufferView {
    /// Родительский буфер, содержащий данные.
    parent: MatrixBufferHandle,
    /// Смещение начала представления в элементах f32 относительно начала родителя.
    offset_elements: usize,
    /// Длина представления в элементах f32.
    len_elements: usize,
    /// Количество строк представления (обычно равно `len_elements` для вектора-столбца).
    rows: usize,
    /// Количество столбцов представления (обычно 1).
    cols: usize,
}

impl MatrixBufferView {
    /// Создаёт новое представление над частью родительского буфера.
    ///
    /// По умолчанию представление интерпретируется как вектор-столбец
    /// размером `len_elements × 1`.
    ///
    /// # Аргументы
    /// * `parent` – дескриптор родительского буфера.
    /// * `offset_elements` – смещение в элементах f32 от начала родителя.
    /// * `len_elements` – длина представления в элементах f32.
    ///
    /// # Паника
    /// Паникует, если `offset_elements + len_elements` выходит за пределы
    /// родительского буфера (проверка выполняется при обращении к данным).
    pub fn new(parent: MatrixBufferHandle, offset_elements: usize, len_elements: usize) -> Self {
        assert!(
            offset_elements
                .checked_add(len_elements)
                .map(|end| end <= parent.rows() * parent.cols())
                .unwrap_or(false),
            "MatrixBufferView::new: range [{}, {}) out of bounds for parent of size {}",
            offset_elements,
            offset_elements + len_elements,
            parent.rows() * parent.cols()
        );
        Self {
            parent,
            offset_elements,
            len_elements,
            rows: len_elements,
            cols: 1,
        }
    }

    /// Создаёт представление с явно заданными размерами матрицы.
    ///
    /// # Аргументы
    /// * `rows` – количество строк представления.
    /// * `cols` – количество столбцов представления.
    ///
    /// Должно выполняться `rows * cols == len_elements`.
    pub fn with_shape(
        parent: MatrixBufferHandle,
        offset_elements: usize,
        len_elements: usize,
        rows: usize,
        cols: usize,
    ) -> Self {
        assert_eq!(
            rows * cols,
            len_elements,
            "MatrixBufferView::with_shape: rows*cols must equal len_elements"
        );
        let mut view = Self::new(parent, offset_elements, len_elements);
        view.rows = rows;
        view.cols = cols;
        view
    }

    /// Возвращает количество строк представления.
    #[inline]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Возвращает количество столбцов представления.
    #[inline]
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Возвращает длину представления в элементах f32.
    #[inline]
    pub fn len(&self) -> usize {
        self.len_elements
    }

    /// Возвращает `true`, если представление не содержит элементов.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len_elements == 0
    }

    /// Возвращает родительский дескриптор буфера.
    #[inline]
    pub fn parent_handle(&self) -> &MatrixBufferHandle {
        &self.parent
    }

    /// Возвращает смещение относительно начала родительского буфера (в элементах).
    #[inline]
    pub fn offset_elements(&self) -> usize {
        self.offset_elements
    }

    /// Проверяет, находятся ли данные представления на GPU.
    #[inline]
    pub fn is_gpu(&self) -> bool {
        self.parent.is_gpu()
    }

    /// Возвращает тип устройства памяти, на котором находятся данные.
    #[inline]
    pub fn device_kind(&self) -> MemoryDeviceKind {
        self.parent.device_kind()
    }

    /// Читает диапазон элементов внутри представления.
    ///
    /// Выполняет копирование данных из родительского буфера с учётом
    /// смещения представления. Поддерживает CPU-буферы; для GPU/SSD
    /// поведение зависит от реализации родительского [`MatrixBufferHandle::read_range`].
    ///
    /// # Аргументы
    /// * `start_in_view` – смещение относительно начала представления.
    /// * `len` – количество элементов для чтения.
    ///
    /// # Паника
    /// Паникует, если запрошенный диапазон выходит за пределы представления.
    pub fn read_range(&self, start_in_view: usize, len: usize) -> Vec<f32> {
        assert!(
            start_in_view + len <= self.len_elements,
            "MatrixBufferView::read_range: range out of view bounds"
        );
        self.parent
            .read_range(self.offset_elements + start_in_view, len)
    }

    /// Читает всё содержимое представления.
    pub fn read_all(&self) -> Vec<f32> {
        self.read_range(0, self.len_elements)
    }

    /// Записывает данные в диапазон внутри представления.
    ///
    /// Выполняет запись в родительский буфер с учётом смещения представления.
    /// Поддерживает CPU-буферы; для GPU/SSD поведение зависит от реализации
    /// родительского [`MatrixBufferHandle::write_range`].
    ///
    /// # Аргументы
    /// * `start_in_view` – смещение относительно начала представления.
    /// * `data` – данные для записи; длина должна точно соответствовать диапазону.
    ///
    /// # Паника
    /// Паникует, если запрошенный диапазон выходит за пределы представления.
    pub fn write_range(&self, start_in_view: usize, data: &[f32]) {
        assert!(
            start_in_view + data.len() <= self.len_elements,
            "MatrixBufferView::write_range: range out of view bounds"
        );
        self.parent
            .write_range(self.offset_elements + start_in_view, data);
    }

    /// Записывает данные во всё представление.
    pub fn write_all(&self, data: &[f32]) {
        assert_eq!(
            data.len(),
            self.len_elements,
            "MatrixBufferView::write_all: data length mismatch"
        );
        self.write_range(0, data);
    }
}