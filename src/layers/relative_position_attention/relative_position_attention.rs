// src/layers/relative_position_attention/relative_position_attention.rs

use crate::layers::UniversalLayer;

/// Слой RelativePositionAttention.
///
/// Одноголовое внимание с относительным позиционным смещением.
/// Вход: `(batch, seq_len * d_model)`.
/// Выход: `(batch, seq_len * d_model)`.
///
/// # Формула
///
/// Для каждого элемента последовательности выполняется линейное
/// преобразование Q, K, V:
///
///   q = x · Wq + bq    (batch, seq, d_model)
///   k = x · Wk + bk    (batch, seq, d_model)
///   v = x · Wv + bv    (batch, seq, d_model)
///
/// Затем вычисляется матрица сходства с относительным смещением:
///
///   dot[t, s]   = (q_t · k_s) / sqrt(d_model)
///   score[t, s] = dot[t, s] + rel_bias[s − t + (seq − 1)]
///   weights     = softmax_s(score)
///
/// Результат внимания:
///
///   attn_out[t, i] = Σ_s weights[t, s] · v[s, i]
///
/// и выходной линейный слой:
///
///   y[t, j] = Σ_i attn_out[t, i] · Wo[j, i] + bo[j]
///
/// # Параметры
///
/// Раскладка параметров в общем буфере (смещение `slice.start`):
///
/// | Смещение        | Размер             | Что                    |
/// |-----------------|--------------------|------------------------|
/// | `0`             | `d_model · d_model`| `Wq` (row-major)       |
/// | `d²`            | `d_model`          | `bq`                   |
/// | `d² + d`        | `d_model · d_model`| `Wk` (row-major)       |
/// | `2·d² + d`      | `d_model`          | `bk`                   |
/// | `2·d² + 2·d`    | `d_model · d_model`| `Wv` (row-major)       |
/// | `3·d² + 2·d`    | `d_model`          | `bv`                   |
/// | `3·d² + 3·d`    | `d_model · d_model`| `Wo` (row-major)       |
/// | `4·d² + 3·d`    | `d_model`          | `bo`                   |
/// | `4·d² + 4·d`    | `2·seq_len − 1`    | `relative_bias`        |
///
/// Итого: `4·(d_model² + d_model) + (2·seq_len − 1)` параметров.
///
/// # Состояние forward
///
/// Кэш прямого прохода (Q, K, V, weights, attn_out) **не хранится**
/// в структуре слоя. Он создаётся per-chunk в `forward_buffered` из
/// `TempMatrixPool` и передаётся в backward через
/// `BufferedContext::RelativePositionAttention`.
///
/// Это делает слой безопасным для чанкового распараллеливания: каждый
/// чанк получает свои изолированные state-буферы, гонки за общее
/// состояние слоя не возникает.
///
/// Матрица `scores` в контекст не кладётся — она используется только
/// внутри forward для вычисления softmax и освобождается сразу после.
pub struct RelativePositionAttention {
    /// Длина последовательности.
    pub seq_len: usize,
    /// Размерность модели (число признаков на одном токене).
    pub d_model: usize,
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