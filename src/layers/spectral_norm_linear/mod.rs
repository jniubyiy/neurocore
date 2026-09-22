// src/layers/spectral_norm_linear/mod.rs

pub mod spectral_norm_linear;

pub mod gpu;   // публичный, но реализация вызывает panic

/// Градиентный адаптер слоя SpectrallyNormalizedLinear
/// (MIGRATION_PLAN.md §7, Фаза 5+).
///
/// LARS-adaptive нормировка градиента по W с использованием
/// **эффективной спектральной нормы** `W_sn`, а не сырых `‖W‖`.
/// См. документацию модуля `adapter` для полного вывода формулы.
pub mod adapter;

mod cpu;

pub use spectral_norm_linear::SpectrallyNormalizedLinear;