// src/compute_manager/distributor_v2/topology.rs
//
// Снимок топологии v2.
//
// Содержит всё, что нужно `SmartDistributor::strategy` для выбора
// оператора и `SmartDistributor::dispatch` для подготовки job'а:
//   * есть ли GPU и под каким id;
//   * сколько CPU-воркеров;
//   * ссылка на `MemoryExecutor` (для `ensure_local`);
//   * ссылка на `GpuCompute` (для вложения в OptimizerStepJob).
//
// Снимок иммутабелен в течение одной эпохи: `SmartDistributor` клонирует
// его при каждом `dispatch` (все поля — `Arc` или `Copy`, клонирование
// дешёвое) и не удерживает блокировку на время работы оператора.

use std::sync::{Arc, RwLock};

use crate::compute_manager::core::device_spec::DeviceId;
use crate::compute_manager::operators_v2::gpu_v2::GpuCompute;
use crate::compute_manager::jobs_v2::OperatorKind;
use crate::compute_manager::operators_v2::memory_v2::types::MemoryDeviceKind;
use crate::compute_manager::operators_v2::memory_v2::MemoryExecutor;

/// Снимок доступных ресурсов.
#[derive(Clone)]
pub struct TopologySnapshot {
    /// Доступен ли GPU в текущей конфигурации.
    pub has_gpu: bool,

    /// Идентификатор GPU-устройства, если `has_gpu == true`.
    pub gpu_device_id: Option<usize>,

    /// Количество вычислительных CPU-воркеров.
    pub cpu_workers: usize,

    /// Разделяемый `MemoryExecutor`.
    pub memory_executor: Arc<RwLock<MemoryExecutor>>,

    /// `GpuCompute`, если GPU доступен.
    pub gpu_compute: Option<Arc<GpuCompute>>,

    /// Номер текущей эпохи (увеличивается `on_epoch_boundary`).
    pub epoch: usize,
}

impl TopologySnapshot {
    /// Возвращает устройство памяти для CPU-сегмента.
    #[inline]
    pub fn memory_kind_for_cpu(&self) -> MemoryDeviceKind {
        MemoryDeviceKind::HostRam
    }

    /// Возвращает устройство памяти для GPU-сегмента.
    /// `None`, если GPU недоступен.
    #[inline]
    pub fn memory_kind_for_gpu(&self) -> Option<MemoryDeviceKind> {
        self.gpu_device_id
            .map(|id| MemoryDeviceKind::DeviceVram(DeviceId(id)))
    }

    /// Возвращает устройство памяти, соответствующее оператору.
    ///
    /// Для `OperatorKind::Memory` возвращает `HostRam` (у Memory-оператора
    /// нет собственного хранилища — он перемещает данные между устройствами).
    #[inline]
    pub fn memory_kind_for_operator(
        &self,
        op: OperatorKind,
    ) -> Option<MemoryDeviceKind> {
        match op {
            OperatorKind::Cpu => Some(self.memory_kind_for_cpu()),
            OperatorKind::Gpu => self.memory_kind_for_gpu(),
            OperatorKind::Memory => Some(self.memory_kind_for_cpu()),
        }
    }
}