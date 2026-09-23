// src/layers/multi_resolution_kan_linear/mod.rs

pub mod multi_resolution_kan_linear;

pub mod gpu;   // публичный, но реализация вызывает panic

/// Градиентный адаптер слоя MultiResolutionKANLinear
/// (MIGRATION_PLAN.md §7, Фаза 5+).
///
/// Выравнивает эффективные скорости обучения шести разнородных групп
/// параметров KAN-слоя (bias / mix_logits / mix_temp / spline_coarse /
/// spline_fine / base_weight) и делает норму per-sample градиента
/// инвариантной к размеру батча. См. документацию модуля `adapter`.
pub mod adapter;

mod cpu;

pub use multi_resolution_kan_linear::MultiResolutionKANLinear;
