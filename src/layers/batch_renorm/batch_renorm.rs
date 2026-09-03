// src/layers/batch_renorm/batch_renorm.rs

use crate::layers::UniversalLayer;

/// Слой BatchRenorm1d — улучшенный BatchNorm с обучаемыми поправками r и d.
///
/// Формула:
/// y = (x - μ_B) / σ_B * r * γ + (d * γ + β),
/// где μ_B, σ_B — статистики текущего батча,
/// r и d — обучаемые параметры коррекции (инициализируются 1 и 0),
/// γ и β — обычные параметры BatchNorm.
/// Во время инференса используются скользящие средние running_mean и running_var.
pub struct BatchRenorm1d {
    /// Количество признаков (столбцов матрицы).
    pub features: usize,
    /// Обучаемый параметр r (вектор длины features).
    /// Инициализируется 1.0, но инициализацию мы делегируем ParamStore.
    /// Здесь только количество параметров.
    /// Все обучаемые параметры: γ, β, r, d — 4 * features.
    /// Running статистики хранятся отдельно в состоянии слоя (не в ParamStore).
}

impl BatchRenorm1d {
    /// Создаёт новый слой с заданным числом признаков.
    ///
    /// # Паника
    /// Паникует, если `features == 0`.
    pub fn new(features: usize) -> Self {
        assert!(features > 0, "BatchRenorm1d: features must be positive");
        Self { features }
    }
}

impl UniversalLayer for BatchRenorm1d {
    fn as_batch_renorm(&self) -> Option<&BatchRenorm1d> {
        Some(self)
    }

    fn param_len(&self) -> usize {
        4 * self.features // γ, β, r, d
    }

    fn input_features(&self) -> usize {
        self.features
    }

    fn output_features(&self) -> usize {
        self.features
    }
}