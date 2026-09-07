// src/layers/adaptive_dropout/mod.rs

pub mod adaptive_dropout;
pub mod gpu;
mod cpu;

pub use adaptive_dropout::AdaptiveDropout;