// src/compute_manager/jobs_v2/mod.rs
//
// Корневой модуль заданий (jobs) архитектуры v2.
//
// Типы jobs — базовый словарь общения между графом и распределителем.
// После этапа B модуль подключается к crate безусловно; `allow(dead_code)`
// сохранён, потому что часть публичного API (`Job::kind()`,
// `JobResult::is_failed` и т.п.) может не вызываться из текущего кода,
// но является частью контракта.

#![allow(dead_code, unused_imports)]

pub mod types;

pub use types::{
    // Routing / identification
    Job,
    JobHandle,
    JobKind,
    JobResult,
    OperatorKind,

    // Contexts container
    ForwardContextsV2,

    // Payloads
    BackwardSegmentJob,
    ConnectorDirection,
    ConnectorOpJob,
    ConnectorOpKind,
    DimOpJob,
    DimOpKind,
    ForwardSegmentJob,
    LossJob,
    MigrateJob,
    OptimizerModifyGradsJob,
    OptimizerApplyUpdateJob,
    ParamInitJob,
};