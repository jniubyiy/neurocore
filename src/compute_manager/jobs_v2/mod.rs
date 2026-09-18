// src/compute_manager/jobs_v2/mod.rs
//
// Корневой модуль заданий (jobs) архитектуры v2.
//
// Всё содержимое модуля не используется старым кодом и не вызывается из
// `run_training!`. Модуль изолирован фичей `v2` (см.
// `src/compute_manager/mod.rs`) и подключается к основному пути только на
// финальном этапе миграции.

#![cfg_attr(feature = "v2", allow(dead_code, unused_imports))]

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
    OptimizerStepJob,
    ParamInitJob,
};