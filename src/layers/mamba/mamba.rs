// src/layers/mamba/mamba.rs

use crate::layers::UniversalLayer;

/// Упрощённый слой Mamba (State Space Model).
///
/// Реализует дискретизированное уравнение состояния:
///   h_t = A_bar * h_{t-1} + B_bar * x_t
///   y_t = C * h_t + D * x_t
/// где `A_bar = exp(Δ A)`, `B_bar = Δ B` (упрощение).
///
/// Параметры:
/// - A: матрица `state_dim × state_dim`
/// - B: матрица `state_dim × input_dim`
/// - C: матрица `input_dim × state_dim`
/// - D: скаляр
/// - Δ: скаляр (шаг дискретизации)
///
/// Вход: `(batch, seq_len * input_dim)`. Выход: `(batch, seq_len * input_dim)`.
///
/// # Состояние forward
///
/// Кэш прямого прохода (`h_all`, `a_bar`, `b_bar`) **не хранится**
/// в структуре слоя. Он создаётся per-chunk в `forward_buffered` из
/// `TempMatrixPool` и передаётся в backward через
/// `BufferedContext::Mamba { input, h_all, a_bar, b_bar }`.
///
/// Это делает слой безопасным для чанкового распараллеливания:
/// каждый чанк получает свои изолированные state-буферы, гонки
/// за общее состояние слоя не возникает.
///
/// # Раскладка state-буферов
///
/// - `h_all`: `(batch * seq_len × state_dim)`, column-major,
///   элемент `(r, t, i)` лежит по адресу `i * (batch * seq) + (r * seq + t)`.
/// - `a_bar`: `(state_dim × state_dim)`, row-major.
/// - `b_bar`: `(state_dim × input_dim)`, row-major.
pub struct Mamba {
    pub seq_len: usize,
    pub input_dim: usize,
    pub state_dim: usize,
}

impl Mamba {
    /// Создаёт новый слой.
    ///
    /// # Паника
    /// Паникует, если какой-либо из размеров равен нулю.
    pub fn new(seq_len: usize, input_dim: usize, state_dim: usize) -> Self {
        assert!(seq_len > 0 && input_dim > 0 && state_dim > 0);
        Self {
            seq_len,
            input_dim,
            state_dim,
        }
    }
}

impl UniversalLayer for Mamba {
    fn as_mamba(&self) -> Option<&Mamba> {
        Some(self)
    }

    fn param_len(&self) -> usize {
        let n = self.state_dim;
        let d = self.input_dim;
        // A: n*n, B: n*d, C: d*n, D: 1, delta: 1
        n * n + n * d + d * n + 2
    }

    fn input_features(&self) -> usize {
        self.seq_len * self.input_dim
    }

    fn output_features(&self) -> usize {
        self.seq_len * self.input_dim
    }
}