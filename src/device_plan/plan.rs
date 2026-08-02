// src/device_plan/plan.rs

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use crate::compute_manager::device_spec::DeviceSpec;
use crate::compute_manager::gpu::init::GpuContext;
use crate::compute_manager::memory_executor::MemoryExecutor;

// ---------------------------------------------------------------------------
// Устройства (Compute / Storage)
// ---------------------------------------------------------------------------

/// Вычислительное устройство.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ComputeDevice {
    Cpu { id: usize, threads: usize },
    Gpu { id: usize },
}

/// Устройство хранения.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StorageDevice {
    Ram { id: usize, max_mb: u64 },
    Vram { id: usize, gpu_id: usize, max_mb: u64 },
    Ssd { id: usize, path: PathBuf, max_mb: u64 },
}

// ---------------------------------------------------------------------------
// План конфигурации
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct DevicePlan {
    pub compute_devices: Vec<ComputeDevice>,
    pub storage_devices: Vec<StorageDevice>,
    pub default_compute_id: usize,
}

impl DevicePlan {
    /// Создаёт пустой план (без устройств).
    pub fn empty() -> Self {
        Self {
            compute_devices: Vec::new(),
            storage_devices: Vec::new(),
            default_compute_id: 0,
        }
    }

    /// Стандартный план для разработки: CPU id=0 (2 потока, RAM 8 ГБ) + RAM id=0.
    pub fn default() -> Self {
        Self {
            compute_devices: vec![ComputeDevice::Cpu { id: 0, threads: 2 }],
            storage_devices: vec![StorageDevice::Ram { id: 0, max_mb: 8192 }],
            default_compute_id: 0,
        }
    }

    // ---------- Builder методы ----------

    /// Добавляет CPU.
    pub fn cpu(mut self, id: usize, threads: usize) -> Self {
        self.compute_devices.push(ComputeDevice::Cpu { id, threads });
        if self.compute_devices.len() == 1 {
            self.default_compute_id = id;
        }
        self
    }

    /// Добавляет GPU.
    pub fn gpu(mut self, id: usize) -> Self {
        self.compute_devices.push(ComputeDevice::Gpu { id });
        if self.compute_devices.len() == 1 {
            self.default_compute_id = id;
        }
        self
    }

    /// Добавляет RAM-хранилище.
    pub fn ram(mut self, id: usize, max_mb: u64) -> Self {
        self.storage_devices.push(StorageDevice::Ram { id, max_mb });
        self
    }

    /// Добавляет VRAM-хранилище (привязано к GPU).
    pub fn vram(mut self, id: usize, gpu_id: usize, max_mb: u64) -> Self {
        self.storage_devices.push(StorageDevice::Vram { id, gpu_id, max_mb });
        self
    }

    /// Добавляет SSD-хранилище.
    pub fn ssd(mut self, id: usize, path: impl Into<PathBuf>, max_mb: u64) -> Self {
        self.storage_devices.push(StorageDevice::Ssd {
            id,
            path: path.into(),
            max_mb,
        });
        self
    }

    // ---------- Геттеры ----------

    pub fn default_compute_id(&self) -> usize {
        self.default_compute_id
    }

    /// Интерпретатор конфигурационной строки (для удобства).
    ///
    /// Формат: `"cpu:<id>:<threads>; gpu:<id>; ram:<id>:<mb>; vram:<id>:<gpu_id>:<mb>; ssd:<id>:<path>:<mb>"`
    /// Элементы разделяются `;`, пробелы игнорируются.
    pub fn from_config_string(config: &str) -> Result<Self, String> {
        let mut plan = DevicePlan::empty();
        let cleaned = config.replace(' ', "").replace('\t', "");
        if cleaned.is_empty() {
            return Ok(DevicePlan::default());
        }
        let tokens: Vec<&str> = cleaned.split(';').filter(|s| !s.is_empty()).collect();
        for token in tokens {
            if token.starts_with("cpu:") {
                let rest = &token[4..];
                let parts: Vec<&str> = rest.splitn(2, ':').collect();
                if parts.len() != 2 {
                    return Err(format!("Неверный формат CPU: {}", token));
                }
                let id: usize = parts[0].parse().map_err(|_| format!("Неверный ID CPU в '{}'", token))?;
                let threads: usize = parts[1].parse().map_err(|_| format!("Неверные потоки в '{}'", token))?;
                plan = plan.cpu(id, threads);
            } else if token.starts_with("gpu:") {
                let id: usize = token[4..].parse().map_err(|_| format!("Неверный ID GPU в '{}'", token))?;
                plan = plan.gpu(id);
            } else if token.starts_with("ram:") {
                let rest = &token[4..];
                let parts: Vec<&str> = rest.splitn(2, ':').collect();
                if parts.len() != 2 {
                    return Err(format!("Неверный формат RAM: {}", token));
                }
                let id: usize = parts[0].parse().map_err(|_| format!("Неверный ID RAM в '{}'", token))?;
                let mb: u64 = parts[1].parse().map_err(|_| format!("Неверный размер RAM в '{}'", token))?;
                plan = plan.ram(id, mb);
            } else if token.starts_with("vram:") {
                let rest = &token[5..];
                let parts: Vec<&str> = rest.splitn(3, ':').collect();
                if parts.len() != 3 {
                    return Err(format!("Неверный формат VRAM: {}", token));
                }
                let id: usize = parts[0].parse().map_err(|_| format!("Неверный ID VRAM в '{}'", token))?;
                let gpu_id: usize = parts[1].parse().map_err(|_| format!("Неверный GPU ID в '{}'", token))?;
                let mb: u64 = parts[2].parse().map_err(|_| format!("Неверный размер VRAM в '{}'", token))?;
                plan = plan.vram(id, gpu_id, mb);
            } else if token.starts_with("ssd:") {
                let rest = &token[4..];
                if let Some(last_colon) = rest.rfind(':') {
                    let path_and_id = &rest[..last_colon];
                    let mb_str = &rest[last_colon + 1..];
                    let mb: u64 = mb_str.parse().map_err(|_| format!("Неверный размер SSD в '{}'", token))?;
                    // разделяем путь и id: первый ':' отделяет id от пути
                    if let Some(first_colon) = path_and_id.find(':') {
                        let id_str = &path_and_id[..first_colon];
                        let path = &path_and_id[first_colon + 1..];
                        let id: usize = id_str.parse().map_err(|_| format!("Неверный ID SSD в '{}'", token))?;
                        plan = plan.ssd(id, PathBuf::from(path), mb);
                    } else {
                        return Err(format!("Неверный формат SSD (ожидается 'ssd:id:путь:ёмкость'): {}", token));
                    }
                } else {
                    return Err(format!("Неверный формат SSD: {}", token));
                }
            } else {
                return Err(format!("Неизвестный тип устройства: '{}'", token));
            }
        }
        Ok(plan)
    }

    /// Создаёт `MemoryExecutor` и возвращает GPU-контекст, если есть.
    pub fn build_memory_executor(&self) -> (Arc<Mutex<MemoryExecutor>>, Option<Arc<GpuContext>>) {
        let mem_exec = Arc::new(Mutex::new(MemoryExecutor::new()));

        // Регистрация хранилищ RAM и SSD
        for storage in &self.storage_devices {
            match storage {
                StorageDevice::Ram { id, max_mb } => {
                    let spec = DeviceSpec::cpu(*id, *max_mb, 1);
                    mem_exec.lock().unwrap().register_compute_device(spec, None);
                }
                StorageDevice::Ssd { path, max_mb, .. } => {
                    let max_bytes = *max_mb * 1024 * 1024;
                    mem_exec
                        .lock()
                        .unwrap()
                        .register_ssd_cache(path.clone(), max_bytes)
                        .expect("SSD registration failed");
                }
                _ => {}
            }
        }

        // GPU контекст для первого GPU
        let gpu_ctx = self.compute_devices.iter().find_map(|d| match d {
            ComputeDevice::Gpu { id } => {
                let ctx = crate::compute_manager::gpu::init::create_gpu_context(*id)
                    .expect("GPU context creation failed");
                let ctx = Arc::new(ctx);

                // Регистрируем все VRAM для этого GPU
                for storage in &self.storage_devices {
                    if let StorageDevice::Vram { gpu_id, max_mb, .. } = storage {
                        if *gpu_id == *id {
                            let spec = DeviceSpec::gpu(*id, *max_mb, false);
                            mem_exec.lock().unwrap().register_compute_device(spec, Some(ctx.clone()));
                        }
                    }
                }
                Some(ctx)
            }
            _ => None,
        });

        (mem_exec, gpu_ctx)
    }
}