// src/layers/adaptive_dropout/adaptive_dropout.rs

use std::sync::RwLock;
use crate::layers::UniversalLayer;

/// Состояние слоя AdaptiveDropout для одного шага forward/backward.
///
/// `mask` содержит бинарную маску `z ∈ {0, 1}` того же размера, что и
/// входной тензор (`batch * features` элементов в column-major).
/// `arg` содержит аргументы сигмоиды `a = (|x| - θ) / T` для каждого элемента.
/// `valid` устанавливается в `true` при выполнении прямого прохода и
/// используется в обратном проходе как индикатор наличия актуального
/// состояния. Пока `valid == false`, содержимое `mask` и `arg` не определено.
pub(crate) struct AdaptiveDropoutState {
    pub mask: Vec<f32>,
    pub arg: Vec<f32>,
    pub valid: bool,
}

/// Слой AdaptiveDropout — dropout с обучаемыми порогом и температурой,
/// зависящими от величины активации.
///
/// Вероятность удержания элемента вычисляется как:
///   p = sigmoid( (|x| - θ) / T ),
/// где θ и T — обучаемые векторы длины `features`.
///
/// Во время прямого прохода генерируется бинарная маска `z ~ Bernoulli(p)`
/// и выход масштабируется: `y = x * z / (p + eps)`.
/// Для обратного прохода сохраняются маска `z` и аргумент `a` (в state).
pub struct AdaptiveDropout {
    pub features: usize,
    /// Зерно для генератора случайных чисел. Используется для воспроизводимости.
    pub seed: u64,
    /// Состояние последнего прямого прохода.
    pub(crate) state: RwLock<AdaptiveDropoutState>,
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
            state: RwLock::new(AdaptiveDropoutState {
                mask: Vec::new(),
                arg: Vec::new(),
                valid: false,
            }),
        }
    }

    /// Сохраняет результаты прямого прохода (маску `z` и аргумент `a`).
    pub(crate) fn store_state(&self, mask: Vec<f32>, arg: Vec<f32>) {
        let mut guard = self.state.write().unwrap();
        guard.mask = mask;
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

    /// Возвращает `true`, если в слое сохранено актуальное состояние
    /// (был выполнен forward и ещё не было invalidate).
    pub(crate) fn has_valid_state(&self) -> bool {
        self.state.read().unwrap().valid
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