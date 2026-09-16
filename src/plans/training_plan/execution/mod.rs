// src/plans/training_plan/execution/mod.rs

//! Оркестрация обучения согласно `TrainingPlan`.
//!
//! Публичный API модуля:
//! * [`execute`] — запускает обучение в отдельном потоке с увеличенным
//!   стеком;
//! * [`TrainingResult`] — итоговый результат обучения.
//!
//! Внутренние подмодули:
//! * `debug`     — отладочные переключатели, константы и утилиты вывода;
//! * `nan_debug` — детектор и репортер NaN/Inf;
//! * `params`    — безопасное чтение параметров и per-sample MSE;
//! * `lsf_evo`   — диагностика эволюции β/θ слоя LearnableSoftplus;
//! * `types`     — `TrainingResult`, `BatchInfo`;
//! * `overrides` — layer-aware канонические инициализации;
//! * `execute`   — оркестраторы `execute` / `execute_inner`.

mod debug;
mod nan_debug;
mod params;
mod lsf_evo;
mod types;
mod overrides;
mod execute;

pub use execute::execute;
pub use types::TrainingResult;