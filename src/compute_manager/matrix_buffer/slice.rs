// src/compute_manager/matrix_buffer/slice.rs

use crate::compute_manager::matrix_buffer::handle::MatrixBufferHandle;

/// Представление непрерывного диапазона строк CPU-буфера.
#[derive(Clone)]
pub struct MatrixBufferSlice {
    parent: MatrixBufferHandle,
    start_row: usize,
    num_rows: usize,
}

impl MatrixBufferSlice {
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

    #[inline]
    pub fn rows(&self) -> usize {
        self.num_rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.parent.cols()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.num_rows * self.cols()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.num_rows == 0
    }

    #[inline]
    pub fn parent(&self) -> &MatrixBufferHandle {
        &self.parent
    }

    #[inline]
    pub fn start_row(&self) -> usize {
        self.start_row
    }

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

    pub fn to_handle(&self) -> MatrixBufferHandle {
        let memory = self.parent.memory();
        let mut mem = memory.write().unwrap();
        let handle = mem
            .acquire_matrix_handle(
                self.num_rows,
                self.cols(),
                crate::compute_manager::memory_executor::types::MemoryDeviceKind::HostRam,
                crate::compute_manager::memory_executor::policy::BufferPriority::Medium,
            )
            .expect("Failed to allocate buffer for slice");
        drop(mem);

        let data = self.read();
        handle.write_range(0, &data);
        handle
    }
}