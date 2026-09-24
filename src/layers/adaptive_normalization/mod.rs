// src/layers/adaptive_normalization/mod.rs

pub mod adaptive_normalization;

/// Градиентный адаптер слоя AdaptiveNormalization
/// (MIGRATION_PLAN.md §7, Фаза 5+).
///
/// Поддерживает 9 режимов (выбираются через `NEUROCORE_ADNORM_MODE`),
/// каждый из которых оптимизирует градиент по своему. См. документацию
/// модуля `adapter` для полного описания формул и env-переменных.
pub mod adapter;

pub mod gpu;   // делаем модуль gpu публичным

mod cpu;

pub use adaptive_normalization::AdaptiveNormalization;