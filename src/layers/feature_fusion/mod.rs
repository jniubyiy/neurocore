// src/layers/feature_fusion/mod.rs

pub mod feature_fusion;

pub mod gpu;   // делаем модуль gpu публичным

mod cpu;

pub use feature_fusion::FeatureFusion;