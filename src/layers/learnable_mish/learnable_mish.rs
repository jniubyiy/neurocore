// src/layers/learnable_mish/learnable_mish.rs

use crate::layers::UniversalLayer;

/// Слой LearnableMish — Mish с обучаемым параметром сглаживания λ.
///
/// Формула: y = x * tanh(λ * softplus(x)), где softplus(x) = ln(1 + e^x).
/// При λ = 1 совпадает с обычным Mish.
/// Параметр λ хранится как скаляр (один на весь слой) и обучается.
pub struct LearnableMish {
    /// Количество признаков (столбцов матрицы). Используется для проверки входной размерности.
    pub features: usize,
}

impl LearnableMish {
    /// Создаёт новый слой с заданным числом признаков.
    ///
    /// # Паника
    /// Паникует, если `features == 0`.
    pub fn new(features: usize) -> Self {
        assert!(features > 0, "LearnableMish: features must be positive");
        Self { features }
    }
}

impl UniversalLayer for LearnableMish {
    fn as_learnable_mish(&self) -> Option<&LearnableMish> {
        Some(self)
    }

    fn param_len(&self) -> usize {
        1 // один обучаемый параметр λ
    }

    fn input_features(&self) -> usize {
        self.features
    }

    fn output_features(&self) -> usize {
        self.features
    }
}