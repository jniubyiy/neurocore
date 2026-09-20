// src/layers/linear/linear.rs

use crate::layers::UniversalLayer;
use crate::layers::adapter::GradientAdapter;

use super::adapter::LinearAdapter;

pub struct Linear {
    pub(crate) in_features: usize,
    pub(crate) out_features: usize,

    /// Пилотный no-op градиентный адаптер (MIGRATION_PLAN.md §7, Фаза 5).
    ///
    /// В Фазе 5 адаптер ничего не модифицирует — только валидирует
    /// среду адаптеров. GPU-версия заморожена до готовности
    /// инфраструктуры для адаптеров любых слоёв.
    pub(crate) adapter: LinearAdapter,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        Self {
            in_features,
            out_features,
            adapter: LinearAdapter::new(),
        }
    }
}

impl UniversalLayer for Linear {
    fn as_linear(&self) -> Option<&Linear> {
        Some(self)
    }

    /// Возвращает пилотный no-op адаптер (MIGRATION_PLAN.md §7, Фаза 5).
    ///
    /// Вызывается в `GraphV2::adapter_pass` между
    /// `optimizer_modify_grads` и `optimizer_apply_update`.
    /// Адаптер не модифицирует градиент — только валидирует среду.
    #[inline]
    fn adapter(&self) -> Option<&dyn GradientAdapter> {
        Some(&self.adapter)
    }

    fn param_len(&self) -> usize {
        self.in_features * self.out_features + self.out_features
    }

    fn input_features(&self) -> usize {
        self.in_features
    }

    fn output_features(&self) -> usize {
        self.out_features
    }
}