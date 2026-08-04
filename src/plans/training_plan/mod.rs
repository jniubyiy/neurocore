// src/training_plan/mod.rs

pub mod plan;
pub mod execution;
pub mod data;
pub mod macros;
pub mod profiling;

pub use plan::{TrainingPlan, Initializer, ValidationConfig};
pub use execution::TrainingResult;
pub use profiling::{ProfileMode, ProfileResult};
pub use crate::run_training;