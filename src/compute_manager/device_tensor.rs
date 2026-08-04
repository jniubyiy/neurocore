// src/compute_manager/device_tensor.rs

use faer::Mat;
use crate::compute_manager::memory_executor::{MemoryExecutor, TensorBufferId};
use crate::compute_manager::memory_executor::types::MemoryDeviceKind;
use crate::compute_manager::memory_executor::BufferPriority;
use crate::compute_manager::device_spec::DeviceId;

/// Тензор, который может находиться на CPU (в виде faer::Mat) или на GPU
/// (в виде буфера, управляемого MemoryExecutor).
#[derive(Debug)]
pub enum DeviceTensor {
    Cpu(Mat<f32>),
    Gpu {
        buffer_id: TensorBufferId,
        rows: usize,
        cols: usize,
    },
}

impl DeviceTensor {
    pub fn from_cpu(mat: Mat<f32>) -> Self {
        DeviceTensor::Cpu(mat)
    }

    pub fn from_gpu(buffer_id: TensorBufferId, rows: usize, cols: usize) -> Self {
        DeviceTensor::Gpu { buffer_id, rows, cols }
    }

    pub fn rows(&self) -> usize {
        match self {
            DeviceTensor::Cpu(mat) => mat.nrows(),
            DeviceTensor::Gpu { rows, .. } => *rows,
        }
    }

    pub fn cols(&self) -> usize {
        match self {
            DeviceTensor::Cpu(mat) => mat.ncols(),
            DeviceTensor::Gpu { cols, .. } => *cols,
        }
    }

    pub fn is_cpu(&self) -> bool {
        matches!(self, DeviceTensor::Cpu(_))
    }

    pub fn is_gpu(&self) -> bool {
        matches!(self, DeviceTensor::Gpu { .. })
    }

    /// Читает данные тензора в матрицу на CPU (копирует).
    pub fn to_cpu(&self, mem_exec: &mut MemoryExecutor) -> Mat<f32> {
        match self {
            DeviceTensor::Cpu(mat) => mat.clone(),
            DeviceTensor::Gpu { buffer_id, rows, cols } => {
                // временно перемещаем в HostRam, читаем, возвращаем обратно
                let target_kind = MemoryDeviceKind::HostRam;
                mem_exec.move_buffer(*buffer_id, target_kind)
                    .expect("Failed to move GPU buffer to HostRam for reading");
                let resolved = mem_exec.resolve_buffer(*buffer_id, target_kind)
                    .expect("Failed to resolve buffer");
                let slice = resolved.as_host_slice();
                let total = rows * cols;
                let data = slice[..total].to_vec();
                drop(resolved);
                // возвращаем в VRAM (предполагаем устройство 0 – заглушка)
                let vram_kind = MemoryDeviceKind::DeviceVram(DeviceId(0));
                mem_exec.move_buffer(*buffer_id, vram_kind)
                    .expect("Failed to move buffer back to VRAM");
                Mat::from_fn(*rows, *cols, |r, c| data[r * cols + c])
            }
        }
    }

    /// Перемещает тензор на GPU, возвращает новый DeviceTensor::Gpu.
    pub fn to_gpu(&self, mem_exec: &mut MemoryExecutor, gpu_device_id: DeviceId) -> Self {
        match self {
            DeviceTensor::Cpu(mat) => {
                let rows = mat.nrows();
                let cols = mat.ncols();
                let total = rows * cols;
                // собираем плоский вектор без замыканий на mat
                let mut flat = Vec::with_capacity(total);
                for r in 0..rows {
                    for c in 0..cols {
                        flat.push(mat[(r, c)]);
                    }
                }
                let host_id = mem_exec.allocate(MemoryDeviceKind::HostRam, total, BufferPriority::High)
                    .expect("Failed to allocate host buffer for GPU upload");
                {
                    let mut resolved = mem_exec.resolve_buffer(host_id, MemoryDeviceKind::HostRam)
                        .expect("Failed to resolve host buffer");
                    resolved.as_host_slice_mut().copy_from_slice(&flat);
                }
                mem_exec.move_buffer(host_id, MemoryDeviceKind::DeviceVram(gpu_device_id))
                    .expect("Failed to move buffer to GPU VRAM");
                DeviceTensor::Gpu { buffer_id: host_id, rows, cols }
            }
            DeviceTensor::Gpu { buffer_id, rows, cols } => {
                let total = rows * cols;
                // копируем через host (временное перемещение)
                mem_exec.move_buffer(*buffer_id, MemoryDeviceKind::HostRam)
                    .expect("Failed to move GPU buffer to host for copy");
                let data_vec = {
                    let resolved = mem_exec.resolve_buffer(*buffer_id, MemoryDeviceKind::HostRam)
                        .expect("Failed to resolve buffer");
                    resolved.as_host_slice()[..total].to_vec()
                };
                // возвращаем исходный буфер в VRAM (заглушка: всегда на устройство 0)
                mem_exec.move_buffer(*buffer_id, MemoryDeviceKind::DeviceVram(DeviceId(0)))
                    .expect("Failed to restore original GPU buffer");
                // создаём новый буфер на целевом GPU
                let new_id = mem_exec.allocate(MemoryDeviceKind::DeviceVram(gpu_device_id), total, BufferPriority::High)
                    .expect("Failed to allocate new GPU buffer");
                let host_staging_id = mem_exec.allocate(MemoryDeviceKind::HostRam, total, BufferPriority::High)
                    .expect("Failed to allocate staging buffer");
                {
                    let mut resolved = mem_exec.resolve_buffer(host_staging_id, MemoryDeviceKind::HostRam)
                        .expect("Failed to resolve staging buffer");
                    resolved.as_host_slice_mut().copy_from_slice(&data_vec);
                }
                mem_exec.move_buffer(host_staging_id, MemoryDeviceKind::DeviceVram(gpu_device_id))
                    .expect("Failed to move staging buffer to GPU");
                DeviceTensor::Gpu {
                    buffer_id: host_staging_id,
                    rows: *rows,
                    cols: *cols,
                }
            }
        }
    }

    /// Преобразует тензор в матрицу на CPU, освобождая GPU-буфер, если он был.
    pub fn into_cpu(self, mem_exec: &mut MemoryExecutor) -> Mat<f32> {
        match self {
            DeviceTensor::Cpu(mat) => mat,
            DeviceTensor::Gpu { buffer_id, rows, cols } => {
                mem_exec.move_buffer(buffer_id, MemoryDeviceKind::HostRam)
                    .expect("Failed to move GPU buffer to host");
                let data_vec = {
                    let resolved = mem_exec.resolve_buffer(buffer_id, MemoryDeviceKind::HostRam)
                        .expect("Failed to resolve buffer");
                    let total = rows * cols;
                    resolved.as_host_slice()[..total].to_vec()
                };
                mem_exec.deallocate_buffer(buffer_id)
                    .expect("Failed to deallocate buffer");
                Mat::from_fn(rows, cols, |r, c| data_vec[r * cols + c])
            }
        }
    }

    /// Преобразует тензор в GPU-тензор, освобождая CPU-матрицу, если она была.
    pub fn into_gpu(self, mem_exec: &mut MemoryExecutor, gpu_device_id: DeviceId) -> Self {
        match self {
            DeviceTensor::Cpu(mat) => {
                let rows = mat.nrows();
                let cols = mat.ncols();
                let total = rows * cols;
                let mut flat = Vec::with_capacity(total);
                for r in 0..rows {
                    for c in 0..cols {
                        flat.push(mat[(r, c)]);
                    }
                }
                let host_id = mem_exec.allocate(MemoryDeviceKind::HostRam, total, BufferPriority::High)
                    .expect("Failed to allocate host buffer");
                {
                    let mut resolved = mem_exec.resolve_buffer(host_id, MemoryDeviceKind::HostRam)
                        .expect("Failed to resolve host buffer");
                    resolved.as_host_slice_mut().copy_from_slice(&flat);
                }
                mem_exec.move_buffer(host_id, MemoryDeviceKind::DeviceVram(gpu_device_id))
                    .expect("Failed to move buffer to GPU");
                DeviceTensor::Gpu { buffer_id: host_id, rows, cols }
            }
            gpu_tensor @ DeviceTensor::Gpu { .. } => gpu_tensor,
        }
    }
}