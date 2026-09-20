// src/plans/training_plan/execution/types.rs
//
// Тип результата обучения.

use std::collections::HashMap;

use crate::compute_manager::core::dim_change::DynamicTensor;
use crate::layers::adapter::AdapterSummary;

use super::super::profiling::ProfileResult;

pub struct TrainingResult {
    pub tensors: HashMap<String, DynamicTensor>,
    pub final_loss: f32,
    pub training_time_secs: f64,
    pub best_epoch: usize,
    pub best_loss: f32,
    pub zero_loss_epoch: Option<usize>,
    pub profile: Option<ProfileResult>,
    pub monitor_summary: Option<crate::logging::TrainingSummary>,

    /// Сводка работы градиентных адаптеров (MIGRATION_PLAN.md §7, Фаза 6).
    ///
    /// `Some`, если за обучение был хотя бы один вызов адаптера. `None`,
    /// если ни у одного слоя нет адаптера, либо если адаптеры отключены
    /// через `NEUROCORE_DISABLE_ADAPTERS=1`.
    ///
    /// L2-метрики (`mean_scale` и др.) заполняются только при
    /// `NEUROCORE_DEBUG_ADAPTER=1` (иначе замер не выполняется — см.
    /// `layers::adapter::stats`). Если флага нет, `measured_calls == 0`,
    /// а `report()` содержит пометку «L2 not measured».
    pub adapter_summary: Option<AdapterSummary>,
}