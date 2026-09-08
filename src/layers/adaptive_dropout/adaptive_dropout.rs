// src/layers/adaptive_dropout/adaptive_dropout.rs

use std::sync::Mutex;
use crate::layers::UniversalLayer;

/// Слой AdaptiveDropout — dropout с обучаемыми порогом и температурой,
/// зависящими от величины активации.
///
/// Вероятность удержания элемента вычисляется как:
/// p = sigmoid( (|x| - θ) / T ), где θ и T — обучаемые векторы длины features.
/// Во время прямого прохода генерируется бинарная маска z ~ Bernoulli(p)
/// и выход масштабируется: y = x * z / (p + eps).
/// Для обратного прохода маска сохраняется в слое (Mutex).
pub struct AdaptiveDropout {
    pub features: usize,
    /// Зерно для генератора случайных чисел. Используется для воспроизводимости.
    pub seed: u64,
    /// Маска последнего прямого прохода. Используется только в обратном.
    pub(crate) mask: Mutex<Option<Vec<f32>>>,
}

impl AdaptiveDropout {
    /// Создаёт новый слой с seed = 0 (для обратной совместимости).
    pub fn new(features: usize) -> Self {
        Self::new_with_seed(features, 0)
    }

    /// Создаёт новый слой с заданным seed.
    ///
    /// # Паника
    /// Паникует, если `features == 0`.
    pub fn new_with_seed(features: usize, seed: u64) -> Self {
        assert!(features > 0, "AdaptiveDropout: features must be positive");
        Self {
            features,
            seed,
            mask: Mutex::new(None),
        }
    }
}

impl UniversalLayer for AdaptiveDropout {
    fn as_adaptive_dropout(&self) -> Option<&AdaptiveDropout> {
        Some(self)
    }

    fn param_len(&self) -> usize {
        2 * self.features // θ и T
    }

    fn input_features(&self) -> usize {
        self.features
    }

    fn output_features(&self) -> usize {
        self.features
    }
}