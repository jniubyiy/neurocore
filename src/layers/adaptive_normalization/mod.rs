// src/layers/adaptive_normalization/mod.rs

pub mod adaptive_normalization;

pub mod gpu;   // делаем модуль gpu публичным

mod cpu;

pub use adaptive_normalization::AdaptiveNormalization;