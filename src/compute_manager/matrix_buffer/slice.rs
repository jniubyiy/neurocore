// src/compute_manager/matrix_buffer/slice.rs

use crate::compute_manager::matrix_buffer::handle::MatrixBufferHandle;

/// Представление непрерывного диапазона строк CPU-буфера.
///
/// `MatrixBufferSlice` не владеет данными и не создаёт отдельного буфера.
/// Все операции чтения/записи делегируются родительскому `MatrixBufferHandle`
/// с учётом смещения `start_row`.
///
/// Работает только с CPU-хранилищем, для GPU/SSD использовать нельзя.
#[derive(Clone)]
pub struct MatrixBufferSlice {
    parent: MatrixBufferHandle,
    start_row: usize,
    num_rows: usize,
}

impl MatrixBufferSlice {
    /// Создаёт новый слайс над частью родительского буфера.
    ///
    /// # Аргументы
    /// * `parent` – дескриптор родительского буфера (CPU).
    /// * `start_row` – индекс первой строки слайса.
    /// * `num_rows` – количество строк слайса.
    ///
    /// # Паника
    /// Паникует, если диапазон выходит за пределы родительского буфера.
    pub fn new(parent: MatrixBufferHandle, start_row: usize, num_rows: usize) -> Self {
        assert!(
            start_row + num_rows <= parent.rows(),
            "MatrixBufferSlice::new: range [{}, {}) out of bounds for parent with {} rows",
            start_row,
            start_row + num_rows,
            parent.rows()
        );
        assert!(
            !parent.is_gpu(),
            "MatrixBufferSlice supports only CPU buffers"
        );
        Self {
            parent,
            start_row,
            num_rows,
        }
    }

    /// Возвращает количество строк слайса.
    #[inline]
    pub fn rows(&self) -> usize {
        self.num_rows
    }

    /// Возвращает количество столбцов слайса (равно колонкам родителя).
    #[inline]
    pub fn cols(&self) -> usize {
        self.parent.cols()
    }

    /// Возвращает общее количество элементов в слайсе.
    #[inline]
    pub fn len(&self) -> usize {
        self.num_rows * self.cols()
    }

    /// Возвращает `true`, если слайс не содержит элементов.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.num_rows == 0
    }

    /// Возвращает ссылку на родительский дескриптор.
    #[inline]
    pub fn parent(&self) -> &MatrixBufferHandle {
        &self.parent
    }

    /// Возвращает индекс первой строки слайса в родительском буфере.
    #[inline]
    pub fn start_row(&self) -> usize {
        self.start_row
    }

    /// Читает все данные слайса в плоский вектор (column-major).
    ///
    /// Порядок элементов: сначала все строки первого столбца, затем второго и т.д.
    pub fn read(&self) -> Vec<f32> {
        let rows_total = self.parent.rows();
        let cols = self.cols();
        let start = self.start_row;
        let count = self.num_rows;

        self.parent.with_cpu_data(|data| {
            let mut result = Vec::with_capacity(count * cols);
            for c in 0..cols {
                for r in 0..count {
                    result.push(data[c * rows_total + start + r]);
                }
            }
            result
        })
    }

    /// Записывает данные в слайс.
    ///
    /// # Аргументы
    /// * `data` – плоский вектор в column-major порядке, длина должна быть
    ///   равна `rows() * cols()`.
    ///
    /// # Паника
    /// Паникует, если длина данных не соответствует размеру слайса.
    pub fn write(&self, data: &[f32]) {
        assert_eq!(
            data.len(),
            self.len(),
            "MatrixBufferSlice::write: data length mismatch (expected {}, got {})",
            self.len(),
            data.len()
        );

        let rows_total = self.parent.rows();
        let cols = self.cols();
        let start = self.start_row;
        let count = self.num_rows;

        self.parent.with_cpu_data_mut(|buffer| {
            for c in 0..cols {
                for r in 0..count {
                    buffer[c * rows_total + start + r] = data[c * count + r];
                }
            }
        });
    }

    /// Копирует содержимое слайса в новый `MatrixBufferHandle` (CPU).
    ///
    /// Эта функция полезна, если нужно передать слайс в другой поток,
    /// так как `MatrixBufferSlice` не является `Send` из-за `MatrixBufferHandle`.
    /// Для параллельной обработки чаще используется выделение отдельного буфера.
    pub fn to_handle(&self) -> MatrixBufferHandle {
        let pool = self.parent.memory();
        let mut mem = pool.lock().unwrap();
        let handle = mem
            .acquire_matrix_handle(
                self.num_rows,
                self.cols(),
                crate::compute_manager::memory_executor::types::MemoryDeviceKind::HostRam,
                crate::compute_manager::memory_executor::policy::BufferPriority::Medium,
            )
            .expect("Failed to allocate buffer for slice");
        drop(mem);

        self.write(
            &self.read(), // inefficient, but simple
        );
        handle
    }
}