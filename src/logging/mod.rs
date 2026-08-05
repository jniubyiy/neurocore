// src/logging/mod.rs

pub mod logger;
pub mod panic_logger;
pub mod diagnostics;
pub mod training_monitor;

pub use logger::Logger;
pub use panic_logger::{install_panic_hook, log, register_thread};
pub use diagnostics::{DiagContext, ParamsStats, capture_diagnostics, format_diagnostics_report};
pub use training_monitor::{TrainingMonitor, MonitorConfig, EpochSummary, TrainingSummary, Warning};