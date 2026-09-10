// src/layers/ind_rnn/ind_rnn.rs

use std::sync::RwLock;
use crate::layers::UniversalLayer;

/// Кэш прямого прохода для обратного распространения.
///
/// Все тензоры хранятся в column-major раскладке, согласованной с общей
/// раскладкой проекта.
///
/// - `input` — входной тензор, форма `(batch, seq_len * input_dim)`, column-major.
/// - `hidden_states` — все скрытые состояния, форма `(batch * seq_len, input_dim)`,
///   row-major по индексу `(r * seq_len + t) * input_dim + j`.
///   Такая раскладка выбрана потому, что к скрытым состояниям идёт адресация
///   по индексам (r, t, j) при переборе во времени.
pub(crate) struct IndRNNForwardCache {
    /// Входной тензор в column-major порядке: `(batch, seq_len * input_dim)`.
    pub input: Vec<f32>,
    /// Все скрытые состояния: `(batch * seq_len, input_dim)`, индексация
    /// `(r * seq_len + t) * input_dim + j`.
    pub hidden_states: Vec<f32>,
}

/// Состояние слоя IndRNN.
///
/// Содержит кэш последнего прямого прохода и флаг `valid`, указывающий,
/// актуален ли кэш для предстоящего обратного прохода.
/// Пока `valid == false`, содержимое `cache` не определено.
pub(crate) struct IndRNNForwardState {
    pub cache: IndRNNForwardCache,
    pub valid: bool,
}

/// Слой IndRNN (Independent RNN).
///
/// Каждый скрытый нейрон обновляется независимо:
///   h_t = activation(W x_t + u ⊙ h_{t-1} + b)
/// Активация по умолчанию — ReLU.
///
/// Вход: `(batch, seq_len * input_dim)`. Выход: `(batch, seq_len * input_dim)`.
///
/// Параметры:
/// - W: матрица `input_dim × input_dim`
/// - u: вектор `input_dim` (поэлементное умножение)
/// - b: вектор `input_dim`
/// Общее число параметров = `input_dim² + 2 * input_dim`.
pub struct IndRNN {
    /// Размерность входа и скрытого состояния на каждом шаге.
    pub input_dim: usize,
    /// Длина последовательности.
    pub seq_len: usize,
    /// Состояние слоя: кэш последнего forward.
    pub(crate) state: RwLock<IndRNNForwardState>,
}

impl IndRNN {
    /// Создаёт новый слой.
    ///
    /// # Паника
    /// Паникует, если `input_dim == 0` или `seq_len == 0`.
    pub fn new(input_dim: usize, seq_len: usize) -> Self {
        assert!(input_dim > 0, "IndRNN: input_dim must be positive");
        assert!(seq_len > 0, "IndRNN: seq_len must be positive");
        Self {
            input_dim,
            seq_len,
            state: RwLock::new(IndRNNForwardState {
                cache: IndRNNForwardCache {
                    input: Vec::new(),
                    hidden_states: Vec::new(),
                },
                valid: false,
            }),
        }
    }

    /// Сохраняет кэш прямого прохода.
    pub(crate) fn store_cache(&self, cache: IndRNNForwardCache) {
        let mut guard = self.state.write().unwrap();
        guard.cache = cache;
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

impl UniversalLayer for IndRNN {
    fn as_ind_rnn(&self) -> Option<&IndRNN> {
        Some(self)
    }

    fn param_len(&self) -> usize {
        let d = self.input_dim;
        d * d + 2 * d
    }

    fn input_features(&self) -> usize {
        self.seq_len * self.input_dim
    }

    fn output_features(&self) -> usize {
        self.seq_len * self.input_dim
    }
}