// src/layers/adaptive_activation/mod.rs

pub mod adaptive_activation;

pub mod gpu;   // делаем модуль gpu публичным

mod cpu;

pub use adaptive_activation::AdaptivePerFeatureActivation;