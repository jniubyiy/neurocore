// src/layers/mamba/mamba.rs

use std::sync::Mutex;
use crate::layers::UniversalLayer;

/// Кэш прямого прохода для обратного распространения.
pub(crate) struct MambaForwardCache {
    /// Входной тензор, сохранённый в column-major порядке (batch * seq_len * input_dim).
    pub input: Vec<f32>,
    /// Все скрытые состояния (batch * seq_len * state_dim).
    pub h_all: Vec<f32>,
    /// Дискретизированная матрица A_bar (state_dim x state_dim).
    pub A_bar: Vec<f32>,
    /// Дискретизированная матрица B_bar (state_dim x input_dim).
    pub B_bar: Vec<f32>,
}

/// Упрощённый слой Mamba (State Space Model).
///
/// Реализует дискретизированное уравнение состояния:
/// h_t = A_bar * h_{t-1} + B_bar * x_t
/// y_t = C * h_t + D * x_t
/// где A_bar = exp(Δ A), B_bar = Δ B (упрощение).
///
/// Параметры:
/// - A: матрица state_dim × state_dim
/// - B: матрица state_dim × input_dim
/// - C: вектор state_dim
/// - D: скаляр
/// - Δ: скаляр (шаг дискретизации)
///
/// Вход: (batch, seq_len * input_dim). Выход: (batch, seq_len * input_dim).
pub struct Mamba {
    pub seq_len: usize,
    pub input_dim: usize,
    pub state_dim: usize,
    /// Состояние для хранения кэша прямого прохода (используется в обратном).
    pub(crate) state: Mutex<Option<MambaForwardCache>>,
}

impl Mamba {
    pub fn new(seq_len: usize, input_dim: usize, state_dim: usize) -> Self {
        assert!(seq_len > 0 && input_dim > 0 && state_dim > 0);
        Self {
            seq_len,
            input_dim,
            state_dim,
            state: Mutex::new(None),
        }
    }

    /// Сохраняет кэш прямого прохода.
    pub(crate) fn store_cache(&self, cache: MambaForwardCache) {
        let mut state = self.state.lock().unwrap();
        *state = Some(cache);
    }

    /// Извлекает кэш прямого прохода.
    pub(crate) fn take_cache(&self) -> Option<MambaForwardCache> {
        let mut state = self.state.lock().unwrap();
        state.take()
    }
}

impl UniversalLayer for Mamba {
    fn as_mamba(&self) -> Option<&Mamba> {
        Some(self)
    }

    fn param_len(&self) -> usize {
        let n = self.state_dim;
        let d = self.input_dim;
        // A: n*n, B: n*d, C: n, D: 1, delta: 1
        n * n + n * d + n + 2
    }

    fn input_features(&self) -> usize {
        self.seq_len * self.input_dim
    }

    fn output_features(&self) -> usize {
        self.seq_len * self.input_dim
    }
}