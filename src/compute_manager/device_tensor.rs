// src/compute_manager/device_tensor.rs

use faer::Mat;
use crate::compute_manager::memory_executor::{MemoryExecutor, TensorBufferId};
use crate::compute_manager::memory_executor::types::MemoryDeviceKind;
use crate::compute_manager::memory_executor::BufferPriority;
use crate::compute_manager::device_spec::DeviceId;
use crate::compute_manager::persistent_buffer::DeviceBufferId;

/// Тензор, который может находиться на CPU (в виде faer::Mat) или на GPU
/// (в виде буфера, управляемого MemoryExecutor, либо постоянного (persistent) буфера).
#[derive(Debug)]
pub enum DeviceTensor {
    Cpu(Mat<f32>),
    Gpu {
        buffer_id: TensorBufferId,
        rows: usize,
        cols: usize,
    },
    /// Постоянный GPU‑буфер, выделенный заранее и живущий всю эпоху.
    GpuPersistent {
        buffer: vulkano::buffer::Subbuffer<[f32]>,
        rows: usize,
        cols: usize,
        persistent_id: DeviceBufferId,
    },
}

impl DeviceTensor {
    pub fn from_cpu(mat: Mat<f32>) -> Self {
        DeviceTensor::Cpu(mat)
    }

    pub fn from_gpu(buffer_id: TensorBufferId, rows: usize, cols: usize) -> Self {
        DeviceTensor::Gpu { buffer_id, rows, cols }
    }

    /// Создаёт тензор, ссылающийся на постоянный GPU‑буфер.
    pub fn from_persistent_gpu(
        buffer: vulkano::buffer::Subbuffer<[f32]>,
        rows: usize,
        cols: usize,
        persistent_id: DeviceBufferId,
    ) -> Self {
        DeviceTensor::GpuPersistent {
            buffer,
            rows,
            cols,
            persistent_id,
        }
    }

    /// Создаёт тензор, ссылающийся на постоянный CPU‑буфер (представлен как Mat).
    /// В будущем может быть заменено на прямую работу с CPU persistent, но пока используем Mat.
    pub fn from_persistent_cpu(mat: Mat<f32>) -> Self {
        // Для CPU persistent буферов можно было бы хранить ссылку, но пока Mat достаточно.
        DeviceTensor::Cpu(mat)
    }

    pub fn rows(&self) -> usize {
        match self {
            DeviceTensor::Cpu(mat) => mat.nrows(),
            DeviceTensor::Gpu { rows, .. } => *rows,
            DeviceTensor::GpuPersistent { rows, .. } => *rows,
        }
    }

    pub fn cols(&self) -> usize {
        match self {
            DeviceTensor::Cpu(mat) => mat.ncols(),
            DeviceTensor::Gpu { cols, .. } => *cols,
            DeviceTensor::GpuPersistent { cols, .. } => *cols,
        }
    }

    pub fn is_cpu(&self) -> bool {
        matches!(self, DeviceTensor::Cpu(_))
    }

    pub fn is_gpu(&self) -> bool {
        matches!(self, DeviceTensor::Gpu { .. } | DeviceTensor::GpuPersistent { .. })
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
            DeviceTensor::GpuPersistent { buffer, rows, cols, .. } => {
                // Копируем данные из постоянного GPU‑буфера на CPU через staging.
                // Используем временный буфер.
                let total = rows * cols;
                // Для простоты создадим временный CPU‑буфер через mem_exec, скопируем туда данные.
                let host_kind = MemoryDeviceKind::HostRam;
                let staging_id = mem_exec.allocate(host_kind, total, BufferPriority::High)
                    .expect("Failed to allocate staging buffer");
                // Копируем GPU -> Host через команды Vulkan. Это требует GpuCompute, но здесь у нас только mem_exec.
                // Вместо этого используем более простой подход: читаем напрямую, если буфер host‑visible?
                // Так как persistent буфер может быть device‑local, проще временно вызвать GpuCompute.
                // В данном контексте мы не имеем доступа к GpuCompute, поэтому этот метод может вызываться редко.
                // Для полной реализации нужно передать GpuCompute, но пока оставим заглушку.
                // В идеале DeviceTensor не должен заниматься копированием; это дело GpuCompute.
                // Поэтому для persistent буферов этот метод не будет использоваться напрямую.
                // Вместо этого будем использовать специализированные методы в GpuCompute.
                // Но для обратной совместимости можно запаниковать, указав, что для persistent нужен GpuCompute.
                panic!("to_cpu on GpuPersistent requires GpuCompute; use gpu_compute.read_persistent_buffer instead");
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
            DeviceTensor::GpuPersistent { buffer, rows, cols, persistent_id } => {
                // Перенос persistent буфера на другой GPU невозможен без пересоздания.
                // Вместо этого возвращаем тот же тензор, считая, что устройство не меняется.
                // Или можно создать новый persistent на другом GPU, но это требует изменения DevicePlacement.
                // Пока оставляем без изменений.
                DeviceTensor::GpuPersistent {
                    buffer: buffer.clone(),
                    rows: *rows,
                    cols: *cols,
                    persistent_id: persistent_id.clone(),
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
            DeviceTensor::GpuPersistent { buffer, rows, cols, persistent_id: _ } => {
                // Нельзя просто так освободить persistent буфер, он управляется извне.
                // Поэтому здесь мы не освобождаем, а просто читаем данные.
                // Нужно использовать GpuCompute для чтения. Вызываем panic или реализуем через GpuCompute.
                panic!("into_cpu on GpuPersistent not supported without GpuCompute");
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
            gpu_tensor @ DeviceTensor::Gpu { .. } | gpu_tensor @ DeviceTensor::GpuPersistent { .. } => gpu_tensor,
        }
    }
}