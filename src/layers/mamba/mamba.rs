// src/layers/mamba/mamba.rs

use std::sync::RwLock;
use crate::layers::UniversalLayer;

/// Кэш прямого прохода для обратного распространения.
///
/// Все тензоры хранятся в column-major раскладке, согласованной с общей
/// раскладкой проекта.
///
/// - `input` — входной тензор, форма `(batch, seq_len * input_dim)`, column-major.
/// - `h_all` — все скрытые состояния, форма `(batch * seq_len, state_dim)`,
///   индексация `(r * seq_len + t) * state_dim + i`.
/// - `A_bar` — дискретизированная матрица A, форма `(state_dim, state_dim)`,
///   row-major по индексу `i * state_dim + j` (как в шейдере `mamba_discretize.comp`).
/// - `B_bar` — дискретизированная матрица B, форма `(state_dim, input_dim)`,
///   row-major по индексу `i * input_dim + j`.
pub(crate) struct MambaForwardCache {
    /// Входной тензор в column-major порядке: `(batch, seq_len * input_dim)`.
    pub input: Vec<f32>,
    /// Все скрытые состояния: `(batch * seq_len, state_dim)`.
    pub h_all: Vec<f32>,
    /// Дискретизированная матрица A_bar: `(state_dim, state_dim)`, row-major.
    pub a_bar: Vec<f32>,
    /// Дискретизированная матрица B_bar: `(state_dim, input_dim)`, row-major.
    pub b_bar: Vec<f32>,
}

/// Состояние слоя Mamba.
///
/// Содержит кэш последнего прямого прохода и флаг `valid`, указывающий,
/// актуален ли кэш для предстоящего обратного прохода.
/// Пока `valid == false`, содержимое `cache` не определено.
pub(crate) struct MambaForwardState {
    pub cache: MambaForwardCache,
    pub valid: bool,
}

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
pub struct Mamba {
    pub seq_len: usize,
    pub input_dim: usize,
    pub state_dim: usize,
    /// Состояние слоя: кэш последнего forward.
    pub(crate) state: RwLock<MambaForwardState>,
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
            state: RwLock::new(MambaForwardState {
                cache: MambaForwardCache {
                    input: Vec::new(),
                    h_all: Vec::new(),
                    a_bar: Vec::new(),
                    b_bar: Vec::new(),
                },
                valid: false,
            }),
        }
    }

    /// Сохраняет кэш прямого прохода.
    pub(crate) fn store_cache(&self, cache: MambaForwardCache) {
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