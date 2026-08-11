// src/compute_manager/matrix_buffer/buffer.rs

use std::ops::{Deref, DerefMut};
use std::sync::{Arc, Mutex};
use faer::Mat;
use crate::compute_manager::memory_executor::{MemoryDeviceKind, MemoryError, MemoryExecutor};

/// Владеющая матрица, управляемая `MemoryExecutor`.
///
/// Данные хранятся в column‑major порядке, совместимом с `faer`.
pub struct MatrixBuffer {
    rows: usize,
    cols: usize,
    data: Vec<f32>,             // column‑major
    memory: Arc<Mutex<MemoryExecutor>>,
    freed: bool,
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
    /// Создаёт новый буфер и резервирует память в менеджере.
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
        Ok(Self { rows, cols, data, memory: memory.clone(), freed: false })
    }

    /// Создаёт фиктивный буфер нулевого размера без резервирования памяти.
    pub fn dummy(_pool: &crate::compute_manager::TempMatrixPool) -> Self {
        Self {
            rows: 0,
            cols: 0,
            data: Vec::new(),
            memory: Arc::new(Mutex::new(MemoryExecutor::new())),
            freed: true,
        }
    }

    pub fn rows(&self) -> usize { self.rows }
    pub fn cols(&self) -> usize { self.cols }
    pub fn size(&self) -> usize { self.rows * self.cols }

    /// Возвращает копию данных в виде `faer::Mat<f32>`.
    pub fn to_mat(&self) -> Mat<f32> {
        if self.rows == 0 || self.cols == 0 {
            return Mat::zeros(self.rows, self.cols);
        }
        Mat::from_fn(self.rows, self.cols, |r, c| {
            self.data[c * self.rows + r]
        })
    }

    /// Записывает содержимое матрицы `src` в буфер.
    pub fn copy_from_mat(&mut self, src: &Mat<f32>) {
        assert_eq!(src.nrows(), self.rows);
        assert_eq!(src.ncols(), self.cols);
        for c in 0..self.cols {
            for r in 0..self.rows {
                self.data[c * self.rows + r] = src[(r, c)];
            }
        }
    }

    /// Неизменяемый доступ к матрице (копия).
    pub fn as_mat(&self) -> Mat<f32> {
        self.to_mat()
    }

    /// Мутабельный доступ с автоматической записью обратно при дропе guard'а.
    pub fn as_mat_mut(&mut self) -> MatrixMutGuard<'_> {
        let mat = self.to_mat();
        MatrixMutGuard { buf: self, mat }
    }

    pub fn deallocate(&mut self) {
        if !self.freed {
            let elements = self.rows * self.cols;
            if let Ok(mut mem) = self.memory.lock() {
                mem.release_reserved_memory(MemoryDeviceKind::HostRam, elements);
            }
            self.data.clear();
            self.data.shrink_to_fit();
            self.freed = true;
        }
    }
}

impl Drop for MatrixBuffer {
    fn drop(&mut self) {
        self.deallocate();
    }
}