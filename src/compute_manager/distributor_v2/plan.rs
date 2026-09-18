// src/compute_manager/distributor_v2/plan.rs
//
// План размещения сегментов v2.
//
// План строится заново на каждой границе эпохи (`on_epoch_boundary`)
// на основе `TopologySnapshot` и текущего описания сегментов
// (`SegmentTopologyInfo`). Никакой истории план не помнит — эпоха
// сменилась, план пересчитан.
//
// В отличие от старого `compute_executor::ModelPlacement` (который живёт
// в замороженном модуле), `DistributionPlan` v2:
//   * работает с абстрактным `OperatorKind`, а не с `ComputeDevice`;
//   * не хранит `parameter_storage` — миграциями управляет распределитель
//     через `MemoryOperatorV2`;
//   * индексирован по сегментам (в графе v2 сегмент = единица размещения).

use crate::compute_manager::jobs_v2::OperatorKind;
use crate::compute_manager::operators_v2::memory_v2::buffer::MatrixBufferHandle;
use crate::compute_manager::operators_v2::memory_v2::types::MemoryDeviceKind;

use super::strategy;
use super::topology::TopologySnapshot;

// ============================================================================
// Описание сегмента
// ============================================================================

/// Описание сегмента, передаваемое графом в `on_epoch_boundary`.
///
/// Граф знает про `ParamStore`, `ParamSlice` и handle'ы параметров —
/// распределитель про них не знает. Поэтому вся необходимая информация
/// приходит сюда явными полями.
///
/// Поля `grad_handle` и `opt_state_handle` добавлены для корректной миграции
/// всех буферов сегмента одним проходом: если мигрировать только `params`,
/// то при работе на GPU `grads` останутся на CPU, и следующий backward
/// упадёт на assert `grad_params_handle.is_gpu()`.
pub struct SegmentTopologyInfo {
    /// Индекс сегмента в графе.
    pub index: usize,

    /// Суммарное число параметров (для оценки «тяжести»).
    pub param_count: usize,

    /// Дескриптор буфера параметров сегмента.
    /// `None` для сегментов без параметров (коннекторы, dim-op).
    pub param_handle: Option<MatrixBufferHandle>,

    /// Дескриптор буфера градиентов параметров сегмента.
    /// `None` для сегментов без параметров.
    pub grad_handle: Option<MatrixBufferHandle>,

    /// Дескриптор буфера состояния оптимизатора сегмента.
    /// `None`, если состояние ещё не выделено или оптимизатор без состояния.
    pub opt_state_handle: Option<MatrixBufferHandle>,

    /// Где сейчас физически лежат параметры сегмента.
    /// Нужно, чтобы понять, требуется ли миграция.
    pub current_param_location: MemoryDeviceKind,
}

// ============================================================================
// Размещение
// ============================================================================

/// Размещение одного сегмента.
#[derive(Debug, Clone, Copy)]
pub struct SegmentPlacement {
    /// Оператор, исполняющий сегмент.
    pub compute: OperatorKind,

    /// Целевое устройство хранения параметров сегмента.
    pub storage: MemoryDeviceKind,
}

/// План размещения на текущую эпоху.
#[derive(Clone)]
pub struct DistributionPlan {
    /// Номер эпохи, для которой план построен.
    pub epoch: usize,

    /// Размещение каждого сегмента (индекс = индекс сегмента в графе).
    pub placements: Vec<SegmentPlacement>,
}

impl DistributionPlan {
    /// Строит план из снимка топологии и описания сегментов.
    ///
    /// Для каждого сегмента:
    ///   1. Определяет вычислительного оператора через `strategy::select_operator`.
    ///      Но `select_operator` работает с `Job`, а у нас — только описание.
    ///      Поэтому вызывается облегчённая эвристика `select_segment_operator`.
    ///   2. Определяет устройство хранения (VRAM для GPU-сегмента, RAM для CPU).
    pub fn build(
        snapshot: &TopologySnapshot,
        segments: &[SegmentTopologyInfo],
    ) -> Self {
        let mut placements = Vec::with_capacity(segments.len());
        for seg in segments {
            let compute = select_segment_operator(seg, snapshot);
            let storage = snapshot
                .memory_kind_for_operator(compute)
                .unwrap_or(MemoryDeviceKind::HostRam);
            placements.push(SegmentPlacement { compute, storage });
        }
        Self {
            epoch: snapshot.epoch,
            placements,
        }
    }

    /// Возвращает размещение сегмента по индексу.
    #[inline]
    pub fn placement_for(&self, seg_idx: usize) -> Option<&SegmentPlacement> {
        self.placements.get(seg_idx)
    }
}

// ---------------------------------------------------------------------------
// Эвристика выбора оператора для сегмента (без Job)
// ---------------------------------------------------------------------------
//
// `strategy::select_operator` требует `&Job`, но на этапе планирования
// job'а ещё нет. Здесь используется та же логика, но на основании только
// `SegmentTopologyInfo::param_count`.

fn select_segment_operator(
    seg: &SegmentTopologyInfo,
    snapshot: &TopologySnapshot,
) -> OperatorKind {
    if !snapshot.has_gpu {
        return OperatorKind::Cpu;
    }
    if seg.param_count > strategy::HEAVY_THRESHOLD {
        OperatorKind::Gpu
    } else {
        OperatorKind::Cpu
    }
}