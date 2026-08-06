// src/compute_manager/gpu/param_store.rs

use std::io::{self, Write};
use std::sync::{Arc, Mutex};
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};

use crate::compute_manager::device_spec::DeviceId;
use crate::compute_manager::memory_executor::MemoryExecutor;
use super::compute::GpuCompute;

pub struct GpuParamStore {
    pub params: Subbuffer<[f32]>,
    pub grads: Subbuffer<[f32]>,
    pub opt_state: Option<Subbuffer<[f32]>>,
    pub num_params: usize,
    // Идентификаторы raw-буферов для учёта в MemoryExecutor
    raw_param_id: Option<crate::compute_manager::memory_executor::executor::RawBufferId>,
    raw_grad_id: Option<crate::compute_manager::memory_executor::executor::RawBufferId>,
    raw_state_id: Option<crate::compute_manager::memory_executor::executor::RawBufferId>,
    // Ссылка на MemoryExecutor для освобождения raw при дропе
    memory_executor: Option<Arc<Mutex<MemoryExecutor>>>,
    device_id: Option<DeviceId>,
}

impl GpuParamStore {
    /// Создаёт хранилище параметров на GPU без регистрации в MemoryExecutor.
    /// Используется для обратной совместимости, когда MemoryExecutor недоступен.
    pub fn from_cpu(
        allocator: Arc<StandardMemoryAllocator>,
        initial_params: &[f32],
        state_size_per_param: usize,
    ) -> Self {
        let num = initial_params.len();
        let host_memory = MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE;

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
            raw_param_id: None,
            raw_grad_id: None,
            raw_state_id: None,
            memory_executor: None,
            device_id: None,
        }
    }

    /// Создаёт хранилище параметров на GPU и регистрирует буферы в MemoryExecutor
    /// для точного учёта занятой памяти.
    pub fn from_cpu_with_executor(
        allocator: Arc<StandardMemoryAllocator>,
        initial_params: &[f32],
        state_size_per_param: usize,
        memory_executor: &Arc<Mutex<MemoryExecutor>>,
        device_id: DeviceId,
    ) -> Self {
        let num = initial_params.len();
        let host_memory = MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE;

        // Параметры
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

        // Градиенты
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

        // Регистрируем буферы как raw в MemoryExecutor
        let (raw_param_id, raw_grad_id, raw_state_id) = {
            let mut exec = memory_executor.lock().unwrap();
            let raw_param = exec.register_raw_buffer(
                device_id,
                (num * std::mem::size_of::<f32>()) as u64,
                host_memory,
            );
            let raw_grad = exec.register_raw_buffer(
                device_id,
                grads_size,
                host_memory,
            );
            let raw_state = opt_state_buf.as_ref().map(|buf| {
                // buf.len() is u64, size_of::<f32>() is usize; cast size_of to u64 before multiplying
                let size_bytes = buf.len() * (std::mem::size_of::<f32>() as u64);
                exec.register_raw_buffer(
                    device_id,
                    size_bytes,
                    host_memory,
                )
            });
            (Some(raw_param), Some(raw_grad), raw_state)
        };

        GpuParamStore {
            params: params_buf,
            grads: grads_buf,
            opt_state: opt_state_buf,
            num_params: num,
            raw_param_id,
            raw_grad_id,
            raw_state_id,
            memory_executor: Some(memory_executor.clone()),
            device_id: Some(device_id),
        }
    }

    /// Читает параметры с GPU в вектор.
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

impl Drop for GpuParamStore {
    fn drop(&mut self) {
        // Разрегистрируем raw-буферы, если они были зарегистрированы
        if let (Some(exec), Some(_dev_id)) = (&self.memory_executor, self.device_id) {
            let mut exec = exec.lock().unwrap();
            if let Some(id) = self.raw_param_id.take() {
                exec.unregister_raw_buffer(id);
            }
            if let Some(id) = self.raw_grad_id.take() {
                exec.unregister_raw_buffer(id);
            }
            if let Some(id) = self.raw_state_id.take() {
                exec.unregister_raw_buffer(id);
            }
        }
    }
}