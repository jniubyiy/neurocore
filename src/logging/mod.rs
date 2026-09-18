// src/logging/mod.rs

pub mod logger;
pub mod panic_logger;
pub mod training_monitor;

pub use logger::Logger;
pub use panic_logger::{install_panic_hook, log, register_thread};
pub use training_monitor::{TrainingMonitor, MonitorConfig, EpochSummary, TrainingSummary, Warning};