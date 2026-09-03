// src/layers/rms_norm_learnable_eps/rms_norm_learnable_eps.rs

use crate::layers::UniversalLayer;

/// Слой RMSNormWithLearnableEpsilon.
///
/// Выполняет RMS-нормализацию с обучаемым параметром epsilon.
/// Формула: y = x / sqrt(mean(x^2) + eps) * gamma.
/// Параметры:
/// - gamma (вектор длины features)
/// - eps   (вектор длины features)
/// Общее количество параметров = 2 * features.
pub struct RMSNormWithLearnableEpsilon {
    /// Количество признаков.
    pub features: usize,
}

impl RMSNormWithLearnableEpsilon {
    /// Создаёт новый слой.
    ///
    /// # Паника
    /// Паникует, если `features == 0`.
    pub fn new(features: usize) -> Self {
        assert!(features > 0, "RMSNormWithLearnableEpsilon: features must be positive");
        Self { features }
    }
}

impl UniversalLayer for RMSNormWithLearnableEpsilon {
    fn as_rms_norm_learnable_eps(&self) -> Option<&RMSNormWithLearnableEpsilon> {
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