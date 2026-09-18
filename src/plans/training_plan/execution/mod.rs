// src/plans/training_plan/execution/mod.rs
//
//! Оркестрация обучения согласно `TrainingPlan`.
//!
//! Публичный API модуля:
//! * [`execute`] — основной оркестратор (v2): `GraphV2` +
//!   `SmartDistributor` + три оператора + `GraphObserverV2`.
//!   Символическое имя `execute` сохранено — `run_training!` резолвит
//!   именно его, поэтому все примеры автоматически идут через v2.
//! * [`execute_v2`] — то же самое, доступно явно (алиас на `execute`).
//! * [`TrainingResult`] — итоговый результат обучения.
//!
//! Внутренние подмодули:
//! * `types`      — `TrainingResult`;
//! * `execute_v2` — v2-оркестратор (`execute_v2.rs`).
//!
//! # История
//!
//! На этапах A и B v1-оркестратор (`execute.rs`) и его вспомогательный
//! `overrides.rs` ещё присутствовали и вызывались явно через `execute_v1`.
//! На этапе C они удалены; остаётся только v2. Канонические per-layer
//! инициализации (`BatchRenorm1d`, `PerFeatureAttention`) выполняются
//! v2-путём через `graph_v2::bridge_v2::build_layer_aware_overrides_v2`.

mod types;
mod execute_v2;

pub use types::TrainingResult;

/// Основной (и единственный) оркестратор обучения — v2.
///
/// Символическое имя `execute` сохранено, чтобы `run_training!` и все
/// примеры продолжали работать без изменений.
pub use execute_v2::execute;

/// Явный алиас на v2-оркестратор для случаев, когда нужен именно
/// «v2-акцент» в имени.
pub use execute_v2::execute as execute_v2;