// src/layers/feature_fusion/feature_fusion.rs

use crate::layers::UniversalLayer;

/// Слой FeatureFusion — обучаемое глобальное агрегирование признаков с softmax-весами.
///
/// Формула: y_j = sum_i (softmax(W_j)_i * x_i) + b_j.
/// Параметры:
/// - логиты W размера (F_out × F_in)
/// - смещения b размера F_out
/// Общее число параметров = F_out * (F_in + 1).
pub struct FeatureFusion {
    pub in_features: usize,
    pub out_features: usize,
}

impl FeatureFusion {
    /// Создаёт слой.
    ///
    /// # Паника
    /// Паникует, если `in_features == 0` или `out_features == 0`.
    pub fn new(in_features: usize, out_features: usize) -> Self {
        assert!(in_features > 0, "FeatureFusion: in_features must be positive");
        assert!(out_features > 0, "FeatureFusion: out_features must be positive");
        Self { in_features, out_features }
    }
}

impl UniversalLayer for FeatureFusion {
    fn as_feature_fusion(&self) -> Option<&FeatureFusion> {
        Some(self)
    }

    fn param_len(&self) -> usize {
        self.out_features * (self.in_features + 1)
    }

    fn input_features(&self) -> usize {
        self.in_features
    }

    fn output_features(&self) -> usize {
        self.out_features
    }
}