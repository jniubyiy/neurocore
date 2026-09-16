// src/plans/training_plan/execution/mod.rs

//! Оркестрация обучения согласно `TrainingPlan`.
//!
//! Публичный API модуля:
//! * [`execute`] — запускает обучение в отдельном потоке с увеличенным
//!   стеком;
//! * [`TrainingResult`] — итоговый результат обучения.
//!
//! Внутренние подмодули:
//! * `types`     — `TrainingResult`;
//! * `overrides` — layer-aware канонические инициализации;
//! * `execute`   — оркестраторы `execute` / `execute_inner`.

mod types;
mod overrides;
mod execute;

pub use execute::execute;
pub use types::TrainingResult;