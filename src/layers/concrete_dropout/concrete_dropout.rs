// src/layers/concrete_dropout/concrete_dropout.rs

use crate::layers::UniversalLayer;

/// Слой ConcreteDropout — dropout с обучаемой вероятностью удержания,
/// основанный на Concrete (Gumbel-Softmax) релаксации Bernoulli.
///
/// Параметры:
/// - `logit_p` (обучаемый скаляр) — логит вероятности удержания `p = sigmoid(logit_p)`.
///
/// Во время прямого прохода генерируется непрерывная маска
///   `z = sigmoid((logit_p + log(u) - log(1 - u)) / τ)`,
/// где `u ~ Uniform(0, 1)`, `τ` — температура.
/// Выход: `y = x * z`.
///
/// Слой предназначен для регуляризации и автоматической настройки силы dropout.
///
/// Состояние прямого прохода (аргументы сигмоиды `a`) не хранится в структуре
/// слоя — оно создаётся per-chunk в `forward_buffered` и передаётся через
/// `BufferedContext::ConcreteDropout`.
pub struct ConcreteDropout {
    /// Температура Gumbel-Softmax. Обычно около 0.1.
    pub temperature: f32,
    /// Зерно для генератора случайных чисел. Используется для воспроизводимости.
    pub seed: u64,
}

impl ConcreteDropout {
    /// Создаёт слой с заданной температурой и seed = 0.
    pub fn new(temperature: f32) -> Self {
        Self::new_with_seed(temperature, 0)
    }

    /// Создаёт слой с заданной температурой и seed.
    ///
    /// # Паника
    /// Паникует, если `temperature <= 0`.
    pub fn new_with_seed(temperature: f32, seed: u64) -> Self {
        assert!(
            temperature > 0.0,
            "ConcreteDropout: temperature must be positive"
        );
        Self {
            temperature,
            seed,
        }
    }
}

impl UniversalLayer for ConcreteDropout {
    fn as_concrete_dropout(&self) -> Option<&ConcreteDropout> {
        Some(self)
    }

    fn param_len(&self) -> usize {
        1 // только logit_p
    }

    fn input_features(&self) -> usize {
        0
    }

    fn output_features(&self) -> usize {
        0
    }
}