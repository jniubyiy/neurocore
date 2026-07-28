// src/compute_manager/gpu/param_store.rs

use std::io::{self, Write};
use std::sync::Arc;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};

use super::compute::GpuCompute;

pub struct GpuParamStore {
    pub params: Subbuffer<[f32]>,
    pub grads: Subbuffer<[f32]>,
    pub opt_state: Option<Subbuffer<[f32]>>,
    pub num_params: usize,
}

impl GpuParamStore {
    pub fn from_cpu(
        allocator: Arc<StandardMemoryAllocator>,
        initial_params: &[f32],
        state_size_per_param: usize,
    ) -> Self {
        let num = initial_params.len();
        let host_memory = MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE;

        // Параметры создаются через from_iter (host‑visible автоматически)
        let params_buf = Buffer::from_iter(
            allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: host_memory,
                ..Default::default()
            },
            initial_params.iter().copied(),
        )
        .expect("GpuParamStore params");

        // Градиенты – выделяем через new_unsized с явным размером в байтах
        let grads_size = (num * std::mem::size_of::<f32>()) as u64;
        let grads_buf = Buffer::new_unsized(
            allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: host_memory,
                ..Default::default()
            },
            grads_size,
        )
        .expect("GpuParamStore grads");

        // Состояние оптимизатора (если нужно)
        let opt_state_buf = if state_size_per_param > 0 {
            let total_state = num * state_size_per_param;
            let state_size = (total_state * std::mem::size_of::<f32>()) as u64;
            Some(
                Buffer::new_unsized(
                    allocator,
                    BufferCreateInfo {
                        usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
                        ..Default::default()
                    },
                    AllocationCreateInfo {
                        memory_type_filter: host_memory,
                        ..Default::default()
                    },
                    state_size,
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

    pub fn to_cpu(&self, _gpu_compute: &GpuCompute) -> Vec<f32> {
        eprintln!(
            "[PARAM] to_cpu: reading {} params directly from host‑visible buffer",
            self.num_params
        );
        io::stderr().flush().unwrap();

        let guard = self.params.read().expect("Failed to read params buffer");
        let data = guard.to_vec();
        eprintln!(
            "[PARAM] to_cpu: read {} floats, first: {:?}",
            data.len(),
            &data[..data.len().min(4)]
        );
        io::stderr().flush().unwrap();
        data
    }
}