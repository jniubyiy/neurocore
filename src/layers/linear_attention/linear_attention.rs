// src/layers/linear_attention/linear_attention.rs

use crate::layers::UniversalLayer;

/// Слой линейного внимания (Linear Attention) со стандартным multi-head
/// (d_head = d_model / max_heads), обучаемым количеством голов и обучаемой
/// добавкой к диагонали внимания.
///
/// # Формула
///
/// Слой содержит `max_heads` независимых head'ов, каждый работает в
/// подпространстве размерности `d_head = d_model / max_heads`. Все head'ы
/// видят весь вход размерности `d_model` и проецируют его в своё
/// подпространство:
///
///   q_h = x · Wq_h + bq_h    (batch, seq, d_head)
///   k_h = x · Wk_h + bk_h    (batch, seq, d_head)
///   v_h = x · Wv_h + bv_h    (batch, seq, d_head)
///
/// Внутри своего подпространства каждая голова вычисляет attention:
///
///   g_{t,s} = φ(q_t) · φ(k_s) + self_bias_h · δ_{t,s}
///   attn_h  = Σ_s g_{t,s} · v_s / Σ_s g_{t,s}    (batch, seq, d_head)
///
/// и проецирует результат обратно в d_model:
///
///   y_h = attn_h · Wo_h + bo_h    (batch, seq, d_model)
///
/// Итоговый выход — взвешенная сумма выходов голов:
///
///   y = Σ_{h=0}^{max_heads-1}  w_h · y_h
///
/// где веса w_h определяются мягким числом голов h_soft:
///
///   w_h = smoothclamp(h_soft − h)
///
/// То есть head 0 всегда активен (пока h_soft ≥ 1), head 1 активен при
/// h_soft > 1, и т.д. Убрать голову = уменьшить h_soft, и первой уходит
/// самая «молодая» (последняя).
///
/// # Обучаемое число голов
///
/// h_soft(h_raw) = min_heads + Σ_{k=1}^{max−min} σ((h_raw − θ_k) / τ),
///   θ_k = k − 0.5, τ = 0.1
///
/// Функция имеет плато на целых значениях h_soft и крутые переходы между
/// ними — «резинка с горкой».
///
/// # Параметры
///
/// Для каждого head h = 0..max_heads−1 (порядок в буфере):
///
/// | Смещение         | Размер            | Что                     |
/// |------------------|-------------------|-------------------------|
/// | `0`              | `d_head · d_model`| `Wq_h` (row-major)      |
/// | `d_head·d_model` | `d_head`          | `bq_h`                  |
/// | `+d_head`        | `d_head · d_model`| `Wk_h` (row-major)      |
/// | `+d_head`        | `d_head`          | `bk_h`                  |
/// | `+d_head`        | `d_head · d_model`| `Wv_h` (row-major)      |
/// | `+d_head`        | `d_head`          | `bv_h`                  |
/// | `+d_head`        | `d_model · d_head`| `Wo_h` (row-major)      |
/// | `+d_model·d_head`| `d_model`         | `bo_h`                  |
/// | `+d_model`       | `1`               | `self_bias_h`           |
///
/// Итого на head: `4 · d_head · d_model + 3 · d_head + d_model + 1`.
/// Плюс один общий скаляр `h_raw` в конце.
///
/// # Состояние forward
///
/// Кэш прямого прохода (per-head q_phi/k_phi/v_raw/kv/z/attn_out/y, а также
/// h_raw и h_soft) **не хранится** в структуре слоя. Он создаётся per-chunk
/// в `forward_buffered` из `TempMatrixPool` и передаётся в backward через
/// `BufferedContext::LinearAttention`.
///
/// Это делает слой безопасным для чанкового распараллеливания: каждый
/// чанк получает свои изолированные state-буферы, гонки за общее состояние
/// слоя не возникает.
///
/// # Паника
///
/// `new` паникует, если `d_model` не делится на `max_heads`.
pub struct LinearAttention {
    pub seq_len: usize,
    pub d_model: usize,
    pub min_heads: usize,
    pub max_heads: usize,
}

impl LinearAttention {
    /// Создаёт слой.
    ///
    /// # Паника
    ///
    /// Паникует, если `seq_len == 0`, `d_model == 0`, `min_heads == 0`,
    /// `max_heads < min_heads` или `d_model` не делится на `max_heads`.
    pub fn new(seq_len: usize, d_model: usize, min_heads: usize, max_heads: usize) -> Self {
        assert!(seq_len > 0, "LinearAttention: seq_len must be positive");
        assert!(d_model > 0, "LinearAttention: d_model must be positive");
        assert!(min_heads >= 1, "LinearAttention: min_heads must be >= 1");
        assert!(
            max_heads >= min_heads,
            "LinearAttention: max_heads must be >= min_heads"
        );
        assert!(
            d_model % max_heads == 0,
            "LinearAttention: d_model ({}) must be divisible by max_heads ({}). \
             Pick max_heads as a divisor of d_model.",
            d_model,
            max_heads
        );
        Self {
            seq_len,
            d_model,
            min_heads,
            max_heads,
        }
    }

    /// Размерность подпространства одной головы.
    #[inline]
    pub(crate) fn d_head(&self) -> usize {
        self.d_model / self.max_heads
    }

    /// Число параметров одной головы (без h_raw).
    #[inline]
    pub(crate) fn head_param_count(&self) -> usize {
        let d = self.d_model;
        let dh = self.d_head();
        4 * dh * d + 3 * dh + d + 1
    }
}

impl UniversalLayer for LinearAttention {
    fn as_linear_attention(&self) -> Option<&LinearAttention> {
        Some(self)
    }

    fn param_len(&self) -> usize {
        self.max_heads * self.head_param_count() + 1
    }

    fn input_features(&self) -> usize {
        self.seq_len * self.d_model
    }

    fn output_features(&self) -> usize {
        self.seq_len * self.d_model
    }
}