// src/layers/concrete_dropout/concrete_dropout.rs

use std::sync::RwLock;
use crate::layers::UniversalLayer;

/// Состояние слоя ConcreteDropout для одного шага forward/backward.
///
/// `arg` содержит аргументы сигмоиды
/// `a = (logit_p + log(u) - log(1 - u)) / temperature`
/// для каждого элемента входа (размер `batch * features`, column-major).
/// `valid` устанавливается в `true` при выполнении прямого прохода и
/// используется в обратном проходе как индикатор наличия актуального
/// состояния. Пока `valid == false`, содержимое `arg` не определено.
pub(crate) struct ConcreteDropoutState {
    pub arg: Vec<f32>,
    pub valid: bool,
}

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
pub struct ConcreteDropout {
    /// Температура Gumbel-Softmax. Обычно около 0.1.
    pub temperature: f32,
    /// Зерно для генератора случайных чисел. Используется для воспроизводимости.
    pub seed: u64,
    /// Состояние последнего прямого прохода.
    pub(crate) state: RwLock<ConcreteDropoutState>,
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
            state: RwLock::new(ConcreteDropoutState {
                arg: Vec::new(),
                valid: false,
            }),
        }
    }

    /// Сохраняет аргументы сигмоиды для обратного прохода.
    pub(crate) fn store_state(&self, arg: Vec<f32>) {
        let mut guard = self.state.write().unwrap();
        guard.arg = arg;
        guard.valid = true;
    }

    /// Помечает состояние как недействительное.
    ///
    /// Используется после успешного завершения обратного прохода, чтобы
    /// повторный backward без нового forward вызывал осмысленную панику.
    /// Данные не освобождаются, чтобы избежать лишних аллокаций.
    pub(crate) fn invalidate(&self) {
        let mut guard = self.state.write().unwrap();
        guard.valid = false;
    }

    /// Возвращает `true`, если в слое сохранено актуальное состояние.
    pub(crate) fn has_valid_state(&self) -> bool {
        self.state.read().unwrap().valid
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