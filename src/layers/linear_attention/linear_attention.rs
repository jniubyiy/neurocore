// src/layers/linear_attention/linear_attention.rs

use crate::layers::UniversalLayer;

/// Слой линейного внимания (Linear Attention) с одним головным механизмом.
///
/// Формула (упрощённая, без нормализации по ключам, но с ELU+1):
/// Attention(Q,K,V) ≈ φ(Q) (φ(K)^T V) / (φ(Q) (φ(K)^T 1) + ε)
///
/// Входная размерность features = seq_len * d_model.
/// Параметры:
/// - W_q, W_k, W_v, W_o размером d_model × d_model
/// - b_q, b_k, b_v, b_o размером d_model
/// Всего 4 * (d_model² + d_model).
pub struct LinearAttention {
    /// Длина последовательности (количество токенов).
    pub seq_len: usize,
    /// Размерность модели (общая для Q, K, V).
    pub d_model: usize,
}

impl LinearAttention {
    /// Создаёт слой.
    ///
    /// # Паника
    /// Паникует, если `seq_len == 0` или `d_model == 0`.
    pub fn new(seq_len: usize, d_model: usize) -> Self {
        assert!(seq_len > 0, "LinearAttention: seq_len must be positive");
        assert!(d_model > 0, "LinearAttention: d_model must be positive");
        Self { seq_len, d_model }
    }
}

impl UniversalLayer for LinearAttention {
    fn as_linear_attention(&self) -> Option<&LinearAttention> {
        Some(self)
    }

    fn param_len(&self) -> usize {
        let d = self.d_model;
        4 * (d * d + d)
    }

    fn input_features(&self) -> usize {
        self.seq_len * self.d_model
    }

    fn output_features(&self) -> usize {
        self.seq_len * self.d_model
    }
}