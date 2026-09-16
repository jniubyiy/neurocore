// src/layers/concrete_dropout/concrete_dropout.rs

use std::sync::atomic::{AtomicU64, Ordering};

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
/// # Воспроизводимость и стохастичность
///
/// Поле `seed` задаёт базовое зерно генератора. Дополнительно слой хранит
/// `call_counter` — атомарный счётчик вызовов forward, единый для CPU- и
/// GPU-путей. Итоговое зерно для конкретного forward-вызова:
///
///   `effective_seed = self.seed.wrapping_add(call_counter)`
///
/// Благодаря этому:
/// * каждый forward использует **свежую независимую** выборку `u`, что
///   критично для dropout-регуляризации;
/// * при одинаковой последовательности forward-вызовов (один и тот же
///   сценарий обучения) маски воспроизводимы от запуска к запуску;
/// * при чанковом распараллеливании атомарный счётчик гарантирует
///   уникальность seed'ов между чанками, и гонок не возникает.
///
/// Без счётчика (прежнее поведение) seed был константой, и все forward'ы
/// получали одну и ту же маску — слой вырождался в фиксированный гейт.
///
/// Состояние прямого прохода (аргументы сигмоиды `a`) не хранится в
/// структуре слоя — оно создаётся per-chunk в `forward_buffered` и
/// передаётся через `BufferedContext::ConcreteDropout`.
pub struct ConcreteDropout {
    /// Температура Gumbel-Softmax. Обычно около 1.0 (см. Maddison et al.
    /// 2016, Jang et al. 2016 — при T << 1 градиент по `logit_p` умирает
    /// из-за насыщения сигмоиды).
    pub temperature: f32,
    /// Базовое зерно для генератора случайных чисел.
    pub seed: u64,
    /// Атомарный счётчик вызовов forward. Используется для генерации
    /// свежего seed'а на каждый forward-проход (см. описание выше).
    pub(crate) call_counter: AtomicU64,
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
            call_counter: AtomicU64::new(0),
        }
    }

    /// Возвращает следующее значение seed'а, атомарно инкрементируя
    /// внутренний счётчик вызовов forward.
    ///
    /// Используется как CPU-путём (`cpu/mod.rs`), так и GPU-путём
    /// (`gpu/processor.rs`), чтобы гарантировать уникальную маску
    /// на каждый forward-проход.
    ///
    /// # Воспроизводимость
    ///
    /// При детерминированной последовательности forward-вызовов
    /// возвращаемые значения воспроизводимы от запуска к запуску
    /// (при одинаковом `self.seed`).
    ///
    /// # Потокобезопасность
    ///
    /// `fetch_add` — атомарная операция, поэтому метод безопасен
    /// при чанковом распараллеливании: разные воркеры получают разные
    /// `call_idx`.
    #[inline]
    pub fn next_seed(&self) -> u64 {
        let call_idx = self.call_counter.fetch_add(1, Ordering::Relaxed);
        self.seed.wrapping_add(call_idx)
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