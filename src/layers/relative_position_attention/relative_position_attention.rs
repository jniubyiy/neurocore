// src/layers/relative_position_attention/relative_position_attention.rs

use std::sync::RwLock;
use crate::layers::UniversalLayer;

/// Кэш промежуточных результатов прямого прохода для обратного распространения.
///
/// Все тензоры хранятся в column-major раскладке, согласованной с общей
/// раскладкой проекта.
///
/// Раскладка тензоров формы `(batch, seq_len * d_model)`, column-major:
///   элемент `(r, t, j)` лежит по адресу `(t * d_model + j) * batch + r`
///
/// Раскладка тензоров формы `(batch, seq_len * seq_len)`, column-major:
///   элемент `(r, t, s)` лежит по адресу `(t * seq_len + s) * batch + r`
pub(crate) struct RelativePositionAttentionCache {
    /// Преобразованные запросы Q.
    /// Форма `(batch, seq_len * d_model)`, column-major.
    pub q: Vec<f32>,
    /// Преобразованные ключи K.
    /// Форма `(batch, seq_len * d_model)`, column-major.
    pub k: Vec<f32>,
    /// Преобразованные значения V.
    /// Форма `(batch, seq_len * d_model)`, column-major.
    pub v: Vec<f32>,
    /// Скоры внимания до softmax.
    /// Форма `(batch, seq_len * seq_len)`, column-major.
    pub scores: Vec<f32>,
    /// Веса внимания после softmax.
    /// Форма `(batch, seq_len * seq_len)`, column-major.
    pub attention_weights: Vec<f32>,
    /// Результат внимания до выходного линейного слоя.
    /// Форма `(batch, seq_len * d_model)`, column-major.
    pub attn_out: Vec<f32>,
    /// Размер батча.
    pub batch: usize,
    /// Длина последовательности.
    pub seq: usize,
    /// Размерность модели.
    pub d_model: usize,
}

/// Состояние слоя RelativePositionAttention.
///
/// Содержит кэш последнего прямого прохода и флаг `valid`, указывающий,
/// актуален ли кэш для предстоящего обратного прохода.
/// Пока `valid == false`, содержимое `cache` не определено.
pub(crate) struct RelativePositionAttentionState {
    pub cache: RelativePositionAttentionCache,
    pub valid: bool,
}

/// Слой RelativePositionAttention.
///
/// Одноголовое внимание с относительным позиционным смещением.
/// Вход: (batch, seq_len * d_model).
/// Слой выполняет линейные преобразования Q, K, V, добавляет к матрице сходства
/// обучаемое смещение, зависящее от относительной позиции, и применяет softmax.
///
/// Параметры:
/// - W_q, W_k, W_v, W_o размером d_model × d_model,
/// - b_q, b_k, b_v, b_o размером d_model,
/// - relative_bias длиной 2 * seq_len - 1 (для относительных позиций).
/// Общее число параметров = 4*(d_model² + d_model) + (2*seq_len - 1).
pub struct RelativePositionAttention {
    pub seq_len: usize,
    pub d_model: usize,
    /// Состояние слоя: кэш последнего forward.
    pub(crate) state: RwLock<RelativePositionAttentionState>,
}

impl RelativePositionAttention {
    /// Создаёт слой.
    ///
    /// # Паника
    /// Паникует, если `seq_len == 0` или `d_model == 0`.
    pub fn new(seq_len: usize, d_model: usize) -> Self {
        assert!(
            seq_len > 0,
            "RelativePositionAttention: seq_len must be positive"
        );
        assert!(
            d_model > 0,
            "RelativePositionAttention: d_model must be positive"
        );
        Self {
            seq_len,
            d_model,
            state: RwLock::new(RelativePositionAttentionState {
                cache: RelativePositionAttentionCache {
                    q: Vec::new(),
                    k: Vec::new(),
                    v: Vec::new(),
                    scores: Vec::new(),
                    attention_weights: Vec::new(),
                    attn_out: Vec::new(),
                    batch: 0,
                    seq: 0,
                    d_model: 0,
                },
                valid: false,
            }),
        }
    }

    /// Сохраняет кэш прямого прохода.
    pub(crate) fn store_cache(&self, cache: RelativePositionAttentionCache) {
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

impl UniversalLayer for RelativePositionAttention {
    fn as_relative_position_attention(&self) -> Option<&RelativePositionAttention> {
        Some(self)
    }

    fn param_len(&self) -> usize {
        let d = self.d_model;
        4 * (d * d + d) + (2 * self.seq_len - 1)
    }

    fn input_features(&self) -> usize {
        self.seq_len * self.d_model
    }

    fn output_features(&self) -> usize {
        self.seq_len * self.d_model
    }
}