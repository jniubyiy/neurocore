// src/layers/mamba/mamba.rs

use crate::layers::UniversalLayer;

/// Упрощённый слой Mamba (State Space Model).
///
/// Реализует дискретизированное уравнение состояния:
/// h_t = A_bar * h_{t-1} + B_bar * x_t
/// y_t = C * h_t + D * x_t
/// где A_bar = exp(Δ A), B_bar = Δ B (упрощение).
///
/// Параметры:
/// - A: матрица state_dim × state_dim (хранится по диагонали? для простоты полная)
/// - B: вектор state_dim (на каждый входной признак) — упрощённо один вектор на все признаки
/// - C: вектор state_dim
/// - D: скаляр
/// - Δ: скаляр (шаг дискретизации)
///
/// Вход: (batch, seq_len * input_dim). Выход: (batch, seq_len * output_dim).
/// В текущей версии input_dim = output_dim.
pub struct Mamba {
    pub seq_len: usize,
    pub input_dim: usize,
    pub state_dim: usize,
}

impl Mamba {
    pub fn new(seq_len: usize, input_dim: usize, state_dim: usize) -> Self {
        assert!(seq_len > 0 && input_dim > 0 && state_dim > 0);
        Self { seq_len, input_dim, state_dim }
    }
}

impl UniversalLayer for Mamba {
    fn as_mamba(&self) -> Option<&Mamba> {
        Some(self)
    }

    fn param_len(&self) -> usize {
        let n = self.state_dim;
        let d = self.input_dim;
        // A: n*n, B: n, C: n, D: 1, delta: 1
        n * n + 2 * n + 2
    }

    fn input_features(&self) -> usize {
        self.seq_len * self.input_dim
    }

    fn output_features(&self) -> usize {
        self.seq_len * self.input_dim
    }
}