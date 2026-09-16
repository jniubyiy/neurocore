// src/plans/training_plan/execution/types.rs
//
// Тип результата обучения.

use std::collections::HashMap;

use crate::compute_manager::dim_change::DynamicTensor;

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
}