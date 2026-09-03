// src/layers/spectral_norm_linear/mod.rs

pub mod spectral_norm_linear;

pub mod gpu;   // публичный, но реализация вызывает panic

mod cpu;

pub use spectral_norm_linear::SpectrallyNormalizedLinear;