// src/layers/adaptive_normalization/adaptive_normalization.rs

use crate::layers::adapter::GradientAdapter;
use crate::layers::UniversalLayer;

use super::adapter::AdaptiveNormAdapter;

/// Слой AdaptiveNormalization.
///
/// Комбинирует LayerNorm, RMSNorm и BatchNorm с обучаемыми весами выбора
/// (логиты) независимо для каждого признака.
/// Параметры:
/// - гамма и бета для LayerNorm
/// - гамма для RMSNorm
/// - гамма и бета для BatchNorm (batch статистики вычисляются на лету)
/// - логиты выбора (3 на признак)
/// Общее количество параметров = 7 * features.
pub struct AdaptiveNormalization {
    /// Количество признаков (столбцов матрицы).
    pub features: usize,

    /// Градиентный адаптер (MIGRATION_PLAN.md §7, Фаза 5+).
    ///
    /// Вызывается в `GraphV2::adapter_pass` строго между
    /// `optimizer_modify_grads` и `optimizer_apply_update` (инвариант I-1).
    /// Адаптер всегда активен (single permanent mode).
    pub(crate) adapter: AdaptiveNormAdapter,
}

impl AdaptiveNormalization {
    /// Создаёт новый слой с заданным числом признаков.
    ///
    /// # Паника
    /// Паникует, если `features == 0`.
    pub fn new(features: usize) -> Self {
        assert!(features > 0, "AdaptiveNormalization: features must be positive");
        Self {
            features,
            adapter: AdaptiveNormAdapter::new(),
        }
    }
}

impl UniversalLayer for AdaptiveNormalization {
    fn as_adaptive_normalization(&self) -> Option<&AdaptiveNormalization> {
        Some(self)
    }

    /// Возвращает градиентный адаптер слоя.
    ///
    /// Вызывается в `GraphV2::adapter_pass` строго между
    /// `optimizer_modify_grads` и `optimizer_apply_update` (инвариант I-1).
    #[inline]
    fn adapter(&self) -> Option<&dyn GradientAdapter> {
        Some(&self.adapter)
    }

    fn param_len(&self) -> usize {
        7 * self.features
    }

    fn input_features(&self) -> usize {
        self.features
    }

    fn output_features(&self) -> usize {
        self.features
    }
}
