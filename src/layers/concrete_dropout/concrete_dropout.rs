// src/layers/concrete_dropout/concrete_dropout.rs

use std::sync::Mutex;
use crate::layers::UniversalLayer;

/// Слой ConcreteDropout — dropout с обучаемой вероятностью удержания,
/// основанный на Concrete (Gumbel-Softmax) релаксации Bernoulli.
///
/// Параметры:
/// - `logit_p` (обучаемый скаляр) — логит вероятности удержания `p = sigmoid(logit_p)`.
/// Во время прямого прохода генерируется непрерывная маска `z = sigmoid((logit_p + log(u) - log(1-u))/τ)`,
/// где `u ~ Uniform(0,1)`, `τ` — температура (фиксированная или передаваемая через `extra`).
/// Выход = вход * z.
///
/// Слой предназначен для регуляризации и автоматической настройки силы dropout.
pub struct ConcreteDropout {
    /// Температура Gumbel-Softmax. Обычно около 0.1.
    pub temperature: f32,
    /// Сохранённый аргумент сигмоиды (для обратного прохода).
    /// В forward сохраняем `arg = (logit_p + log(u) - log(1-u)) / temperature`,
    /// чтобы в backward вычислить производную маски.
    pub(crate) mask_state: Mutex<Option<Vec<f32>>>,
}

impl ConcreteDropout {
    /// Создаёт слой с заданной температурой.
    pub fn new(temperature: f32) -> Self {
        assert!(temperature > 0.0, "ConcreteDropout: temperature must be positive");
        Self {
            temperature,
            mask_state: Mutex::new(None),
        }
    }

    /// Сохраняет аргумент сигмоиды (для обратного прохода).
    pub(crate) fn store_mask(&self, arg: Vec<f32>) {
        let mut state = self.mask_state.lock().unwrap();
        *state = Some(arg);
    }

    /// Извлекает сохранённый аргумент сигмоиды.
    pub(crate) fn take_mask(&self) -> Option<Vec<f32>> {
        let mut state = self.mask_state.lock().unwrap();
        state.take()
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
        0 // не зависит от числа признаков
    }

    fn output_features(&self) -> usize {
        0
    }
}