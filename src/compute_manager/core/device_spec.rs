// src/compute_manager/device_spec.rs

use std::path::PathBuf;
use serde::{Deserialize, Serialize};

/// Уникальный идентификатор устройства в системе (0..N-1)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DeviceId(pub usize);

impl std::fmt::Display for DeviceId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "dev{}", self.0)
    }
}

/// Тип вычислительного устройства
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeviceKind {
    Cpu,
    Gpu,
}

/// Описание возможностей устройства (память, вычислительная мощность, пропускная способность)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceCapabilities {
    /// Общий объём доступной памяти (в мегабайтах) — может использоваться для информации
    pub total_memory_mb: u64,
    /// Пиковая производительность (GFLOPS) для линейной алгебры (оценка)
    pub peak_gflops: f64,
    /// Пропускная способность памяти (ГБ/с)
    pub memory_bandwidth_gbs: f64,
    /// Относительный вес производительности (например, по сравнению с эталонным CPU)
    pub relative_speed: f64,
    /// Поддерживает ли устройство unified memory с хостом (для GPU)
    pub unified_memory: bool,
}

/// Лимиты памяти, которые библиотека может использовать на устройстве
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryLimits {
    /// Максимальный объём RAM/VRAM (в мегабайтах), который библиотека может занять
    pub max_memory_mb: u64,
    /// Для CPU: количество потоков, выделяемых под вычисления (если применимо)
    pub compute_threads: Option<usize>,
    /// Для SSD: путь к директории кэша
    pub cache_path: Option<PathBuf>,
    /// Максимальный объём SSD-кэша (в мегабайтах)
    pub max_cache_mb: Option<u64>,
}

/// Полное описание устройства для регистрации в системе
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceSpec {
    pub id: DeviceId,
    pub kind: DeviceKind,
    pub capabilities: DeviceCapabilities,
    pub limits: MemoryLimits,
}

impl DeviceSpec {
    /// Создать спецификацию CPU с заданными параметрами
    pub fn cpu(id: usize, ram_mb: u64, threads: usize) -> Self {
        DeviceSpec {
            id: DeviceId(id),
            kind: DeviceKind::Cpu,
            capabilities: DeviceCapabilities {
                total_memory_mb: ram_mb,
                peak_gflops: 50.0,       // заглушка, будет уточнено профилированием
                memory_bandwidth_gbs: 30.0,
                relative_speed: 1.0,
                unified_memory: true,    // CPU всегда unified
            },
            limits: MemoryLimits {
                max_memory_mb: ram_mb,
                compute_threads: Some(threads),
                cache_path: None,
                max_cache_mb: None,
            },
        }
    }

    /// Создать спецификацию GPU
    pub fn gpu(id: usize, vram_mb: u64, unified: bool) -> Self {
        DeviceSpec {
            id: DeviceId(id),
            kind: DeviceKind::Gpu,
            capabilities: DeviceCapabilities {
                total_memory_mb: vram_mb,
                peak_gflops: 1000.0,     // заглушка
                memory_bandwidth_gbs: 200.0,
                relative_speed: 10.0,
                unified_memory: unified,
            },
            limits: MemoryLimits {
                max_memory_mb: vram_mb,
                compute_threads: None,
                cache_path: None,
                max_cache_mb: None,
            },
        }
    }

    /// Добавить SSD-кэш к спецификации
    pub fn with_ssd_cache(mut self, path: impl Into<PathBuf>, capacity_mb: u64) -> Self {
        self.limits.cache_path = Some(path.into());
        self.limits.max_cache_mb = Some(capacity_mb);
        self
    }
}