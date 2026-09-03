// src/layers/learnable_softplus/learnable_softplus.rs

use crate::layers::UniversalLayer;

/// Слой LearnableSoftplus — Softplus с обучаемыми параметрами порога θ и масштаба β.
///
/// Формула: y = (1/β) * ln(1 + exp(β * (x - θ))).
/// При β = 1, θ = 0 вырождается в обычный Softplus.
/// Параметры β и θ представлены векторами длины `features` (по одному на признак).
pub struct LearnableSoftplus {
    /// Количество признаков (столбцов матрицы).
    pub features: usize,
}

impl LearnableSoftplus {
    /// Создаёт новый слой с заданным числом признаков.
    ///
    /// # Паника
    /// Паникует, если `features == 0`.
    pub fn new(features: usize) -> Self {
        assert!(features > 0, "LearnableSoftplus: features must be positive");
        Self { features }
    }
}

impl UniversalLayer for LearnableSoftplus {
    fn as_learnable_softplus(&self) -> Option<&LearnableSoftplus> {
        Some(self)
    }

    fn param_len(&self) -> usize {
        2 * self.features
    }

    fn input_features(&self) -> usize {
        self.features
    }

    fn output_features(&self) -> usize {
        self.features
    }
}