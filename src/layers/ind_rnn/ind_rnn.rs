// src/layers/ind_rnn/ind_rnn.rs

use std::sync::Mutex;
use crate::layers::UniversalLayer;

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
    /// Состояние, хранящее скрытые состояния последнего прямого прохода для обратного.
    pub(crate) state: Mutex<Option<Vec<f32>>>,
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