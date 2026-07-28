// src/compute_manager/device_plan.rs

use std::path::PathBuf;
use crate::compute_manager::device::Device;
use crate::compute_manager::device_spec::{DeviceId, DeviceSpec, DeviceKind};
use crate::compute_manager::memory_executor::MemoryExecutor;
use crate::compute_manager::gpu::init::GpuContext;
use std::sync::{Arc, Mutex};

/// План конфигурации устройств и лимитов памяти.
/// Позволяет задать, какие CPU и GPU будут использоваться, и сколько памяти им выделить.
#[derive(Debug, Clone)]
pub struct DevicePlan {
    cpu_threads: usize,
    cpu_ram_mb: u64,
    gpu_id: Option<usize>,
    gpu_vram_mb: Option<u64>,
    ssd_cache_path: Option<PathBuf>,
    ssd_cache_mb: Option<u64>,
}

impl DevicePlan {
    /// Создаёт план с настройками по умолчанию: 2 потока CPU, 8 ГБ RAM, без GPU, без SSD.
    pub fn new() -> Self {
        Self {
            cpu_threads: 2,
            cpu_ram_mb: 8192,
            gpu_id: None,
            gpu_vram_mb: None,
            ssd_cache_path: None,
            ssd_cache_mb: None,
        }
    }

    /// Задаёт параметры CPU: количество потоков и максимальный объём RAM (в МБ).
    pub fn cpu(mut self, threads: usize, ram_mb: u64) -> Self {
        self.cpu_threads = threads;
        self.cpu_ram_mb = ram_mb;
        self
    }

    /// Включает GPU с указанным идентификатором и ограничением VRAM (в МБ).
    pub fn gpu(mut self, id: usize, vram_mb: u64) -> Self {
        self.gpu_id = Some(id);
        self.gpu_vram_mb = Some(vram_mb);
        self
    }

    /// Включает SSD-кэш с путём к директории и максимальным объёмом (в МБ).
    pub fn ssd_cache(mut self, path: impl Into<PathBuf>, capacity_mb: u64) -> Self {
        self.ssd_cache_path = Some(path.into());
        self.ssd_cache_mb = Some(capacity_mb);
        self
    }

    /// Количество потоков CPU.
    pub fn cpu_threads(&self) -> usize {
        self.cpu_threads
    }

    /// Идентификатор GPU, если задан.
    pub fn gpu_id(&self) -> Option<usize> {
        self.gpu_id
    }

    /// Создаёт и конфигурирует `MemoryExecutor` на основе плана, регистрируя устройства с лимитами.
    /// Возвращает `(MemoryExecutor, Option<Arc<GpuContext>>)`, где GpuContext присутствует, если запрошен GPU.
    pub fn build_memory_executor(&self) -> (Arc<Mutex<MemoryExecutor>>, Option<Arc<GpuContext>>) {
        let mem_exec = Arc::new(Mutex::new(MemoryExecutor::new()));

        // Регистрируем CPU
        {
            let mut cpu_spec = DeviceSpec::cpu(0, self.cpu_ram_mb, self.cpu_threads);
            if let Some(ref path) = self.ssd_cache_path {
                cpu_spec = cpu_spec.with_ssd_cache(path.clone(), self.ssd_cache_mb.unwrap_or(0));
            }
            mem_exec.lock().unwrap().register_compute_device(cpu_spec, None);
        }

        // Регистрируем GPU, если задан
        let gpu_ctx = if let Some(gpu_id) = self.gpu_id {
            let vram_mb = self.gpu_vram_mb.unwrap_or(4096);
            let context = crate::compute_manager::gpu::init::create_gpu_context(gpu_id)
                .expect("Failed to create GPU context");
            let context = Arc::new(context);
            let mut gpu_spec = DeviceSpec::gpu(gpu_id, vram_mb, false);
            if let Some(ref path) = self.ssd_cache_path {
                gpu_spec = gpu_spec.with_ssd_cache(path.clone(), self.ssd_cache_mb.unwrap_or(0));
            }
            mem_exec.lock().unwrap().register_compute_device(gpu_spec, Some(context.clone()));
            Some(context)
        } else {
            None
        };

        (mem_exec, gpu_ctx)
    }

    /// Вспомогательный метод для получения `Device` из плана (для обратной совместимости).
    pub fn to_device(&self) -> Device {
        if let Some(gpu_id) = self.gpu_id {
            Device::Gpu { id: gpu_id }
        } else {
            Device::Cpu { threads: self.cpu_threads }
        }
    }
}

impl Default for DevicePlan {
    fn default() -> Self {
        Self::new()
    }
}