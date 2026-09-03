// src/layers/relative_position_attention/relative_position_attention.rs

use crate::layers::UniversalLayer;

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
}

impl RelativePositionAttention {
    pub fn new(seq_len: usize, d_model: usize) -> Self {
        assert!(seq_len > 0, "RelativePositionAttention: seq_len must be positive");
        assert!(d_model > 0, "RelativePositionAttention: d_model must be positive");
        Self { seq_len, d_model }
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