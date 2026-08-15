// src/compute_manager/memory_executor/types.rs

use crate::compute_manager::device_spec::DeviceId;

/// Тип устройства памяти
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryDeviceKind {
    HostRam,
    DeviceVram(DeviceId),
    SsdCache,
}