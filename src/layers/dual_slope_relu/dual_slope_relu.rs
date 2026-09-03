// src/layers/dual_slope_relu/dual_slope_relu.rs

use crate::layers::UniversalLayer;

/// Слой DualSlopeReLU.
///
/// Активация с двумя обучаемыми наклонами:
/// `y = alpha * x` для x < 0 и `y = beta * x` для x >= 0.
/// Каждый наклон представлен вектором длины `features` (по одному на признак).
pub struct DualSlopeReLU {
    /// Количество признаков (столбцов матрицы).
    pub features: usize,
}

impl DualSlopeReLU {
    /// Создаёт новый слой с заданным числом признаков.
    ///
    /// # Паника
    /// Паникует, если `features == 0`.
    pub fn new(features: usize) -> Self {
        assert!(features > 0, "DualSlopeReLU: features must be positive");
        Self { features }
    }
}

impl UniversalLayer for DualSlopeReLU {
    fn as_dual_slope_relu(&self) -> Option<&DualSlopeReLU> {
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