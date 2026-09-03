// src/layers/adaptive_activation/adaptive_activation.rs

use crate::layers::UniversalLayer;

/// Слой AdaptivePerFeatureActivation.
///
/// Хранит обучаемые логиты для выбора комбинации базовых активаций
/// независимо для каждого признака входной матрицы.
pub struct AdaptivePerFeatureActivation {
    /// Количество входных признаков (столбцов матрицы).
    pub in_features: usize,
    /// Количество базовых активаций в наборе.
    pub num_activations: usize,
}

impl AdaptivePerFeatureActivation {
    /// Создаёт новый слой с заданным числом признаков и числом базовых активаций.
    ///
    /// # Паника
    /// Паникует, если `in_features == 0` или `num_activations < 2`.
    pub fn new(in_features: usize, num_activations: usize) -> Self {
        assert!(in_features > 0, "AdaptivePerFeatureActivation: in_features must be positive");
        assert!(
            num_activations >= 2,
            "AdaptivePerFeatureActivation: num_activations must be at least 2"
        );
        Self { in_features, num_activations }
    }
}

impl UniversalLayer for AdaptivePerFeatureActivation {
    fn as_adaptive_activation(&self) -> Option<&AdaptivePerFeatureActivation> {
        Some(self)
    }

    fn param_len(&self) -> usize {
        self.in_features * self.num_activations
    }

    fn input_features(&self) -> usize {
        self.in_features
    }

    fn output_features(&self) -> usize {
        self.in_features
    }
}