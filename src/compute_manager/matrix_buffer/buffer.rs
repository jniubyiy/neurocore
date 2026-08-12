// src/compute_manager/matrix_buffer/buffer.rs

use std::ops::{Deref, DerefMut};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use faer::Mat;
use vulkano::buffer::Subbuffer;

use crate::compute_manager::device_spec::DeviceId;
use crate::compute_manager::memory_executor::executor::RawBufferId;
use crate::compute_manager::memory_executor::{MemoryDeviceKind, MemoryError, MemoryExecutor};

/// Хранилище данных буфера: CPU или GPU.
pub enum BufferStorage {
    /// Данные находятся в оперативной памяти (column‑major).
    Cpu(Vec<f32>),
    /// Данные находятся в видеопамяти GPU.
    Gpu {
        buffer: Subbuffer<[f32]>,
        raw_id: RawBufferId,
        device_id: DeviceId,
    },
}

/// Владеющая матрица, управляемая `MemoryExecutor`.
///
/// Данные хранятся в column‑major порядке, совместимом с `faer`.
/// Поддерживает как CPU‑память (`Vec<f32>`), так и GPU‑память (`Subbuffer<[f32]>`).
pub struct MatrixBuffer {
    rows: usize,
    cols: usize,
    storage: BufferStorage,
    memory: Arc<Mutex<MemoryExecutor>>,
    freed: bool,
    last_used: Instant,
}

/// Временный мутабельный доступ к матрице, гарантирующий обратную запись при дропе.
pub struct MatrixMutGuard<'a> {
    buf: &'a mut MatrixBuffer,
    mat: Mat<f32>,
}

impl<'a> Deref for MatrixMutGuard<'a> {
    type Target = Mat<f32>;
    fn deref(&self) -> &Self::Target {
        &self.mat
    }
}

impl<'a> DerefMut for MatrixMutGuard<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.mat
    }
}

impl<'a> Drop for MatrixMutGuard<'a> {
    fn drop(&mut self) {
        self.buf.copy_from_mat(&self.mat);
    }
}

impl MatrixBuffer {
    /// Создаёт новый буфер в оперативной памяти и резервирует память в менеджере.
    pub fn new(
        memory: &Arc<Mutex<MemoryExecutor>>,
        rows: usize,
        cols: usize,
    ) -> Result<Self, MemoryError> {
        let elements = rows * cols;
        {
            let mut mem = memory.lock().unwrap();
            mem.reserve_memory(MemoryDeviceKind::HostRam, elements)?;
        }
        let data = vec![0.0f32; elements];
        Ok(Self {
            rows,
            cols,
            storage: BufferStorage::Cpu(data),
            memory: memory.clone(),
            freed: false,
            last_used: Instant::now(),
        })
    }

    /// Создаёт новый буфер в видеопамяти указанного GPU.
    ///
    /// Память резервируется через `MemoryExecutor` и регистрируется в реестре сырых буферов.
    /// Пока реализовано через временный пул GPU‑буферов (без возврата в пул).
    /// В будущем будет заменено прямым выделением.
    pub fn new_gpu(
        memory: &Arc<Mutex<MemoryExecutor>>,
        device_id: DeviceId,
        rows: usize,
        cols: usize,
    ) -> Result<Self, MemoryError> {
        let elements = rows * cols;
        let kind = MemoryDeviceKind::DeviceVram(device_id);
        let (buffer, raw_id) = {
            let mut mem = memory.lock().unwrap();
            mem.acquire_temp_buffer(kind, elements)
        };
        Ok(Self {
            rows,
            cols,
            storage: BufferStorage::Gpu {
                buffer,
                raw_id,
                device_id,
            },
            memory: memory.clone(),
            freed: false,
            last_used: Instant::now(),
        })
    }

    /// Создаёт GPU‑буфер из уже существующих компонентов.
    pub fn from_gpu(
        memory: Arc<Mutex<MemoryExecutor>>,
        buffer: Subbuffer<[f32]>,
        raw_id: RawBufferId,
        device_id: DeviceId,
        rows: usize,
        cols: usize,
    ) -> Self {
        Self {
            rows,
            cols,
            storage: BufferStorage::Gpu {
                buffer,
                raw_id,
                device_id,
            },
            memory,
            freed: false,
            last_used: Instant::now(),
        }
    }

    /// Создаёт фиктивный буфер нулевого размера без резервирования памяти.
    pub fn dummy(_pool: &crate::compute_manager::TempMatrixPool) -> Self {
        Self {
            rows: 0,
            cols: 0,
            storage: BufferStorage::Cpu(Vec::new()),
            memory: Arc::new(Mutex::new(MemoryExecutor::new())),
            freed: true,
            last_used: Instant::now(),
        }
    }

    pub fn rows(&self) -> usize { self.rows }
    pub fn cols(&self) -> usize { self.cols }
    pub fn size(&self) -> usize { self.rows * self.cols }

    /// Возвращает true, если данные находятся в видеопамяти GPU.
    pub fn is_gpu(&self) -> bool {
        matches!(self.storage, BufferStorage::Gpu { .. })
    }

    /// Возвращает тип устройства памяти, на котором находится буфер.
    pub fn device_kind(&self) -> MemoryDeviceKind {
        match &self.storage {
            BufferStorage::Cpu(_) => MemoryDeviceKind::HostRam,
            BufferStorage::Gpu { device_id, .. } => MemoryDeviceKind::DeviceVram(*device_id),
        }
    }

    /// Возвращает ссылку на GPU‑буфер, если данные находятся в VRAM.
    pub fn as_gpu_buffer(&self) -> Option<&Subbuffer<[f32]>> {
        match &self.storage {
            BufferStorage::Gpu { buffer, .. } => Some(buffer),
            _ => None,
        }
    }

    /// Возвращает копию данных в виде `faer::Mat<f32>`.
    /// Для GPU‑буфера требуется синхронизация; в текущей версии паникует.
    pub fn to_mat(&self) -> Mat<f32> {
        match &self.storage {
            BufferStorage::Cpu(data) => {
                if self.rows == 0 || self.cols == 0 {
                    return Mat::zeros(self.rows, self.cols);
                }
                Mat::from_fn(self.rows, self.cols, |r, c| {
                    data[c * self.rows + r]
                })
            }
            BufferStorage::Gpu { .. } => {
                panic!("MatrixBuffer::to_mat() is not supported for GPU buffers; use to_cpu() with GpuCompute");
            }
        }
    }

    /// Записывает содержимое матрицы `src` в буфер.
    /// Работает только для CPU‑буфера.
    pub fn copy_from_mat(&mut self, src: &Mat<f32>) {
        match &mut self.storage {
            BufferStorage::Cpu(data) => {
                assert_eq!(src.nrows(), self.rows);
                assert_eq!(src.ncols(), self.cols);
                for c in 0..self.cols {
                    for r in 0..self.rows {
                        data[c * self.rows + r] = src[(r, c)];
                    }
                }
            }
            BufferStorage::Gpu { .. } => {
                panic!("MatrixBuffer::copy_from_mat() is not supported for GPU buffers");
            }
        }
    }

    /// Неизменяемый доступ к матрице (копия). Для GPU паникует.
    pub fn as_mat(&self) -> Mat<f32> {
        self.to_mat()
    }

    /// Мутабельный доступ с автоматической записью обратно при дропе guard'а.
    /// Работает только для CPU‑буфера.
    pub fn as_mat_mut(&mut self) -> MatrixMutGuard<'_> {
        match &self.storage {
            BufferStorage::Cpu(_) => {
                let mat = self.to_mat();
                MatrixMutGuard { buf: self, mat }
            }
            BufferStorage::Gpu { .. } => {
                panic!("MatrixBuffer::as_mat_mut() is not supported for GPU buffers");
            }
        }
    }

    // -------------------------------------------------------------------------
    // Методы прямого доступа к данным (только CPU)
    // -------------------------------------------------------------------------

    /// Возвращает неизменяемую ссылку на внутренние данные (column‑major порядок).
    /// Для GPU‑буфера паникует.
    pub fn as_slice(&self) -> &[f32] {
        match &self.storage {
            BufferStorage::Cpu(data) => data,
            BufferStorage::Gpu { .. } => {
                panic!("MatrixBuffer::as_slice() is not supported for GPU buffers");
            }
        }
    }

    /// Возвращает кортеж: ссылку на данные, количество строк и столбцов.
    pub fn as_slice_with_shape(&self) -> (&[f32], usize, usize) {
        (self.as_slice(), self.rows, self.cols)
    }

    /// Возвращает мутабельную ссылку на внутренние данные.
    /// Для GPU‑буфера паникует.
    pub fn as_slice_mut(&mut self) -> &mut [f32] {
        match &mut self.storage {
            BufferStorage::Cpu(data) => data,
            BufferStorage::Gpu { .. } => {
                panic!("MatrixBuffer::as_slice_mut() is not supported for GPU buffers");
            }
        }
    }

    /// Возвращает кортеж: мутабельную ссылку на данные, количество строк и столбцов.
    pub fn as_mut_slice_with_shape(&mut self) -> (&mut [f32], usize, usize) {
        let rows = self.rows;
        let cols = self.cols;
        let slice = self.as_slice_mut();
        (slice, rows, cols)
    }

    /// Заполняет весь буфер заданным значением. Только CPU.
    pub fn fill(&mut self, value: f32) {
        match &mut self.storage {
            BufferStorage::Cpu(data) => data.fill(value),
            BufferStorage::Gpu { .. } => {
                panic!("MatrixBuffer::fill() is not supported for GPU buffers");
            }
        }
    }

    /// Копирует данные из слайса в буфер. Только CPU.
    pub fn copy_from_slice(&mut self, src: &[f32]) {
        match &mut self.storage {
            BufferStorage::Cpu(data) => {
                assert_eq!(
                    src.len(),
                    data.len(),
                    "MatrixBuffer::copy_from_slice: length mismatch"
                );
                data.copy_from_slice(src);
            }
            BufferStorage::Gpu { .. } => {
                panic!("MatrixBuffer::copy_from_slice() is not supported for GPU buffers");
            }
        }
    }

    /// Изменяет логические размеры буфера, сохраняя общее количество элементов.
    /// Работает для обоих типов хранилища.
    pub fn reshape_into(&mut self, new_rows: usize, new_cols: usize) {
        let total = self.rows * self.cols;
        assert_eq!(
            new_rows * new_cols,
            total,
            "MatrixBuffer::reshape_into: total elements must remain constant"
        );
        self.rows = new_rows;
        self.cols = new_cols;
    }

    /// Возвращает элемент по строке и столбцу (column‑major порядок). Только CPU.
    pub fn get(&self, row: usize, col: usize) -> f32 {
        match &self.storage {
            BufferStorage::Cpu(data) => {
                debug_assert!(row < self.rows && col < self.cols);
                data[col * self.rows + row]
            }
            BufferStorage::Gpu { .. } => {
                panic!("MatrixBuffer::get() is not supported for GPU buffers");
            }
        }
    }

    /// Устанавливает элемент по строке и столбцу. Только CPU.
    pub fn set(&mut self, row: usize, col: usize, value: f32) {
        match &mut self.storage {
            BufferStorage::Cpu(data) => {
                debug_assert!(row < self.rows && col < self.cols);
                data[col * self.rows + row] = value;
            }
            BufferStorage::Gpu { .. } => {
                panic!("MatrixBuffer::set() is not supported for GPU buffers");
            }
        }
    }

    // -------------------------------------------------------------------------
    // Методы для интеграции с TempMatrixPool
    // -------------------------------------------------------------------------

    /// Обновляет метку времени последнего использования.
    pub(crate) fn mark_used(&mut self) {
        self.last_used = Instant::now();
    }

    /// Возвращает время последнего использования.
    pub(crate) fn last_used(&self) -> Instant {
        self.last_used
    }

    // -------------------------------------------------------------------------
    // Освобождение памяти
    // -------------------------------------------------------------------------

    pub fn deallocate(&mut self) {
        if !self.freed {
            match &self.storage {
                BufferStorage::Cpu(data) => {
                    let elements = self.rows * self.cols;
                    if let Ok(mut mem) = self.memory.lock() {
                        mem.release_reserved_memory(MemoryDeviceKind::HostRam, elements);
                    }
                    // данные будут очищены при дропе
                }
                BufferStorage::Gpu { raw_id, .. } => {
                    if let Ok(mut mem) = self.memory.lock() {
                        mem.unregister_raw_buffer(*raw_id);
                    }
                    // Subbuffer будет освобождён при дропе
                }
            }
            self.freed = true;
        }
    }
}

impl Drop for MatrixBuffer {
    fn drop(&mut self) {
        self.deallocate();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute_manager::memory_executor::MemoryExecutor;
    use std::sync::{Arc, Mutex};

    fn create_cpu_buffer() -> (Arc<Mutex<MemoryExecutor>>, MatrixBuffer) {
        let mem = Arc::new(Mutex::new(MemoryExecutor::new()));
        let buf = MatrixBuffer::new(&mem, 3, 4).unwrap();
        (mem, buf)
    }

    #[test]
    fn test_as_slice_cpu() {
        let (_mem, buf) = create_cpu_buffer();
        let slice = buf.as_slice();
        assert_eq!(slice.len(), 12);
        assert!(slice.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_fill_and_get_set_cpu() {
        let (_mem, mut buf) = create_cpu_buffer();
        buf.fill(2.5);
        assert!(buf.as_slice().iter().all(|&x| x == 2.5));

        buf.set(1, 2, 7.0);
        assert_eq!(buf.get(1, 2), 7.0);
        assert_eq!(buf.as_slice()[7], 7.0);
    }

    #[test]
    fn test_copy_from_slice_cpu() {
        let (_mem, mut buf) = create_cpu_buffer();
        let src: Vec<f32> = (0..12).map(|i| i as f32).collect();
        buf.copy_from_slice(&src);
        assert_eq!(buf.as_slice(), &src[..]);
    }

    #[test]
    fn test_reshape_into_cpu() {
        let (_mem, mut buf) = create_cpu_buffer();
        buf.reshape_into(2, 6);
        assert_eq!((buf.rows(), buf.cols()), (2, 6));
        assert_eq!(buf.size(), 12);
    }

    #[test]
    #[should_panic]
    fn test_reshape_invalid_cpu() {
        let (_mem, mut buf) = create_cpu_buffer();
        buf.reshape_into(5, 5);
    }

    #[test]
    fn test_is_gpu_false_for_cpu() {
        let (_mem, buf) = create_cpu_buffer();
        assert!(!buf.is_gpu());
        assert_eq!(buf.device_kind(), MemoryDeviceKind::HostRam);
        assert!(buf.as_gpu_buffer().is_none());
    }
}