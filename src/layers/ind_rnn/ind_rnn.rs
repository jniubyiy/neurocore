// src/layers/ind_rnn/ind_rnn.rs

use std::sync::Mutex;
use crate::layers::UniversalLayer;

/// Кэш прямого прохода для обратного распространения.
pub(crate) struct IndRNNForwardCache {
    /// Входной тензор в column-major порядке (batch * seq_len * input_dim).
    pub input: Vec<f32>,
    /// Все скрытые состояния (batch * seq_len * input_dim).
    pub hidden_states: Vec<f32>,
}

/// Слой IndRNN (Independent RNN).
///
/// Каждый скрытый нейрон обновляется независимо:
/// h_t = activation(W x_t + u ⊙ h_{t-1} + b)
/// Активация по умолчанию — ReLU.
/// Вход: (batch, seq_len * input_dim). Выход: (batch, seq_len * input_dim).
/// Параметры:
/// - W: матрица input_dim × input_dim
/// - u: вектор input_dim (поэлементное умножение)
/// - b: вектор input_dim
/// Общее число параметров = input_dim² + 2 * input_dim.
pub struct IndRNN {
    /// Размерность входа и скрытого состояния на каждом шаге.
    pub input_dim: usize,
    /// Длина последовательности.
    pub seq_len: usize,
    /// Состояние, хранящее вход и скрытые состояния последнего прямого прохода для обратного.
    pub(crate) state: Mutex<Option<IndRNNForwardCache>>,
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
            state: Mutex::new(None),
        }
    }

    /// Сохраняет кэш прямого прохода.
    pub(crate) fn store_cache(&self, cache: IndRNNForwardCache) {
        let mut guard = self.state.lock().unwrap();
        *guard = Some(cache);
    }

    /// Извлекает кэш прямого прохода.
    pub(crate) fn take_cache(&self) -> Option<IndRNNForwardCache> {
        let mut guard = self.state.lock().unwrap();
        guard.take()
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