// src/compute_manager/graph_v2/mod.rs
//
// Корневой модуль графа v2 (GraphV2).
//
// Состав:
//   * types_v2.rs    — SegmentV2, SegmentKindV2, ForwardCacheV2;
//   * model_v2.rs    — GraphV2 (публичный фасад) + конвертации;
//   * builder_v2.rs  — построение сегментов из Vec<LayerDesc>;
//   * forward_v2.rs  — forward через SmartDistributor;
//   * backward_v2.rs — backward через SmartDistributor;
//   * observer_v2.rs — GraphObserverV2, EpochReportV2, MonitorConfigV2;
//   * bridge_v2.rs   — канонические per-layer инициализации.
//
// После этапа B модуль подключается к crate безусловно и является
// основным путём исполнения обучения.

#![allow(dead_code, unused_imports)]

pub mod types_v2;
pub mod builder_v2;
pub mod forward_v2;
pub mod backward_v2;
pub mod model_v2;
pub mod observer_v2;
pub mod bridge_v2;

pub use types_v2::{
    ForwardCacheV2, SegmentForwardStateV2, SegmentKindV2, SegmentV2,
};
pub use model_v2::GraphV2;
pub use observer_v2::{
    EpochReportV2, GraphObserverV2, LossTrendV2, MonitorConfigV2, WarningV2,
};