// src/compute_manager/distributor_v2/mod.rs
//
// Корневой модуль распределителя v2 (SmartDistributor).
//
// Распределитель — единственное место в v2, которое знает про
// существование нескольких операторов (Memory / CPU / GPU). Всё
// остальное (граф, слои, память) видит только его как «чёрный ящик» с
// методом `dispatch(Job) -> JobResult`.
//
// Состав модуля:
//   * topology.rs — TopologySnapshot: снимок доступных ресурсов;
//   * strategy.rs — правила выбора оператора под конкретный JobKind;
//   * plan.rs     — DistributionPlan: какие сегменты куда размещены;
//   * executor.rs — SmartDistributor: dispatch / dispatch_batch /
//                   ensure_local / on_epoch_boundary.
//
// Модуль изолирован фичей `v2` (см. `src/compute_manager/mod.rs`) и
// подключается к основному пути только на финальном этапе миграции.

#![cfg_attr(feature = "v2", allow(dead_code, unused_imports))]

pub mod topology;
pub mod strategy;
pub mod plan;
pub mod executor;

pub use topology::TopologySnapshot;
pub use strategy::select_operator;
pub use plan::{DistributionPlan, SegmentPlacement, SegmentTopologyInfo};
pub use executor::SmartDistributor;