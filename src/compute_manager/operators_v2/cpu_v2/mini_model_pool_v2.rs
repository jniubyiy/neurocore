// src/compute_manager/operators_v2/cpu_v2/mini_model_pool_v2.rs
//
// Алиас на mini-model-хранилище внутри `Scheduler`.
//
// Архитектурная карта v2 предписывает `CpuOperatorV2` содержать
// `Vec<ForwardTimePredictor>` — по одной mini-model на каждый CPU.
//
// Этот вектор **уже существует** в оригинальном `Scheduler`
// (`compute_manager::cpu::scheduler::Scheduler::predictors`), и его
// жизненным циклом управляет именно scheduler (обучение на пороге 50
// замеров, сериализация в `~/.local/share/neurocore/chunk_model_cpuN.json`).
//
// Дублировать это хранилище в `CpuOperatorV2` нельзя: получилось бы две
// независимые модели, обучаемые разными данными, что нарушило бы принцип
// «одна mini-model на один CPU».
//
// Поэтому `MiniModelPoolV2` — не отдельная структура, а документированный
// тип-алиас, фиксирующий, что под «пулом mini-model» в v2 понимается
// ровно `Vec<ForwardTimePredictor>` из `Scheduler`.

use crate::compute_manager::operators_v2::cpu_v2::mini_model::ForwardTimePredictor;

/// Тип-алиас, отражающий содержимое `Scheduler::predictors`.
pub type MiniModelPoolV2 = Vec<ForwardTimePredictor>;