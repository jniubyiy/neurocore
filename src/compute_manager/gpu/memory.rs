// src/compute_manager/gpu/memory.rs

use std::sync::Arc;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use faer::Mat;

pub struct GpuTensor {
    buffer: Subbuffer<[f32]>,
    pub rows: usize,
    pub cols: usize,
}

impl GpuTensor {
    /// Загружает матрицу в GPU‑память.
    pub fn from_matrix(
        mat: &Mat<f32>,
        allocator: &Arc<StandardMemoryAllocator>,
    ) -> Self {
        let rows = mat.nrows();
        let cols = mat.ncols();
        let data: Vec<f32> = (0..rows)
            .flat_map(|r| (0..cols).map(move |c| mat[(r, c)]))
            .collect();

        let buffer = Buffer::from_iter(
            allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            data,
        )
        .expect("Не удалось создать GPU-буфер для тензора");

        GpuTensor { buffer, rows, cols }
    }

    /// Чтение данных с GPU (заглушка).
    pub fn to_matrix(&self) -> Mat<f32> {
        Mat::zeros(self.rows, self.cols)
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Возвращает `Subbuffer<[f32]>` для привязки к дескрипторам.
    pub fn buffer(&self) -> &Subbuffer<[f32]> {
        &self.buffer
    }
}