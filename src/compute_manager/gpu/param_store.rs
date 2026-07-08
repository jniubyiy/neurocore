// src/compute_manager/gpu/param_store.rs

use std::sync::Arc;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};

use super::compute::GpuCompute;

/// Хранилище параметров и состояния оптимизатора в GPU-памяти.
pub struct GpuParamStore {
    /// Буфер параметров модели (все параметры в одном векторе).
    pub params: Subbuffer<[f32]>,
    /// Буфер градиентов (накапливаются во время backward).
    pub grads: Subbuffer<[f32]>,
    /// Буфер состояния оптимизатора (например, для Adam: удвоенная длина).
    pub opt_state: Option<Subbuffer<[f32]>>,
    /// Общее количество параметров.
    pub num_params: usize,
}

impl GpuParamStore {
    /// Создаёт хранилище с начальными параметрами из CPU-вектора.
    pub fn from_cpu(
        allocator: Arc<StandardMemoryAllocator>,
        initial_params: &[f32],
        state_size_per_param: usize, // 0, если состояние не требуется
    ) -> Self {
        let num = initial_params.len();
        let params_buf = Buffer::from_iter(
            allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            initial_params.iter().copied(),
        )
        .expect("GpuParamStore params");

        let grads_buf = Buffer::new_unsized(
            allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
                size: (num * std::mem::size_of::<f32>()) as u64,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            (num * std::mem::size_of::<f32>()) as u64,
        )
        .expect("GpuParamStore grads");

        let opt_state_buf = if state_size_per_param > 0 {
            let total_state = num * state_size_per_param;
            Some(
                Buffer::new_unsized(
                    allocator,
                    BufferCreateInfo {
                        usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
                        size: (total_state * std::mem::size_of::<f32>()) as u64,
                        ..Default::default()
                    },
                    AllocationCreateInfo {
                        memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                        ..Default::default()
                    },
                    (total_state * std::mem::size_of::<f32>()) as u64,
                )
                .expect("GpuParamStore opt_state"),
            )
        } else {
            None
        };

        GpuParamStore {
            params: params_buf,
            grads: grads_buf,
            opt_state: opt_state_buf,
            num_params: num,
        }
    }

    /// Копирует текущие параметры в CPU-вектор (для отладки/сохранения).
    pub fn to_cpu(&self, gpu_compute: &GpuCompute) -> Vec<f32> {
        let staging = gpu_compute.create_buffer(self.num_params, BufferUsage::TRANSFER_DST);
        gpu_compute.copy_buffer_sync(self.params.clone(), staging.clone());
        let data = staging.read().unwrap();
        data.to_vec()
    }
}