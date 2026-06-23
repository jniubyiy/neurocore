// src/compute_manager/device.rs

use std::fmt;

/// Вычислительное устройство
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Device {
    /// Центральный процессор с заданным числом потоков
    Cpu { threads: usize },
    /// Графический процессор с идентификатором (например, номер в системе)
    Gpu { id: usize },
}

impl Default for Device {
    fn default() -> Self {
        Device::Cpu { threads: 2 } // по умолчанию 2 потока CPU
    }
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Device::Cpu { threads } => write!(f, "CPU ({} threads)", threads),
            Device::Gpu { id } => write!(f, "GPU #{}", id),
        }
    }
}

/// Обнаружение доступных вычислительных устройств
pub struct DeviceDetector;

impl DeviceDetector {
    /// Возвращает список всех видимых устройств (CPU + GPU)
    pub fn list_all() -> Vec<Device> {
        let mut devices = Vec::new();

        // CPU всегда доступен
        let logical_cores = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(1);
        devices.push(Device::Cpu {
            threads: logical_cores,
        });

        // GPU: ищем через gpu‑модуль
        if let Some(gpu_list) = crate::compute_manager::gpu::detect_gpus() {
            for (i, _) in gpu_list.iter().enumerate() {
                devices.push(Device::Gpu { id: i });
            }
        }

        devices
    }

    /// Удобный метод: берёт первый CPU с заданным числом потоков (по умолчанию 2)
    pub fn default_cpu() -> Device {
        Device::Cpu { threads: 2 }
    }
}

/// Менеджер вычислений, владеющий выбранным устройством
pub struct ComputeManager {
    pub device: Device,
}

impl ComputeManager {
    pub fn new(device: Device) -> Self {
        Self { device }
    }

    /// Позволяет сменить устройство
    pub fn set_device(&mut self, device: Device) {
        self.device = device;
    }

    /// Возвращает строку с информацией о текущем устройстве
    pub fn info(&self) -> String {
        format!("Compute device: {}", self.device)
    }
}