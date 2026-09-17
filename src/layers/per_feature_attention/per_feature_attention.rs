// src/layers/per_feature_attention/per_feature_attention.rs

use crate::layers::UniversalLayer;

/// Per-feature temporal attention layer.
///
/// # Архитектура
///
/// В отличие от `LinearAttention`, который смешивает информацию между
/// токенами, этот слой работает **по признакам**: каждый входной признак
/// `h ∈ [0, d_model)` получает свою собственную независимую голову,
/// которая обрабатывает **временную последовательность** значений этого
/// признака по всем токенам внутри примера.
///
/// Головы строго упорядочены по индексу признака и не взаимодействуют.
///
/// # Вход / выход
///
/// Вход:  `(batch, seq_len · d_model)` — column-major
/// Выход: `(batch, seq_len · d_model)` — column-major
///
/// # Вычисление одной головы
///
/// Для признака `h` и примера `r` обозначим `x_t = input[r, t, h]`.
/// Голова получает на каждой временной позиции 2-мерный вектор
/// `u_t = [x_t, Δx_t]`, где `Δx_t = x_t − x_{t−1}` (`x_{−1} = 0`). Далее:
///
/// ```text
///   q_raw_t = Wq · u_t + bq       (d_head,)
///   k_raw_t = Wk · u_t + bk
///   v_raw_t = Wv · u_t + bv
///
///   φ(x) = ELU(x) + 1
///
///   kv = Σ_t φ(k_t) ⊗ v_t         (d_head × d_head)
///   z  = Σ_t φ(k_t)               (d_head,)
///
///   attn_t = (self_bias · v_t + φ(q_t) @ kv) / (self_bias + φ(q_t) · z)
///
///   y_t = Wo · attn_t + bo        (скаляр)
/// ```
///
/// # Раскладка параметров
///
/// Для каждой головы `h ∈ [0, d_model)` по порядку:
///
/// | Смещение        | Размер       | Что             |
/// |-----------------|--------------|-----------------|
/// | `0`             | `2 · d_head` | `Wq_h` (row-major) |
/// | `2 · d_head`    | `d_head`     | `bq_h`          |
/// | `3 · d_head`    | `2 · d_head` | `Wk_h` (row-major) |
/// | `5 · d_head`    | `d_head`     | `bk_h`          |
/// | `6 · d_head`    | `2 · d_head` | `Wv_h` (row-major) |
/// | `8 · d_head`    | `d_head`     | `bv_h`          |
/// | `9 · d_head`    | `d_head`     | `Wo_h`          |
/// | `10 · d_head`   | `1`          | `bo_h`          |
/// | `10 · d_head+1` | `1`          | `self_bias_h`   |
///
/// Всего на голову: `10 · d_head + 2`. Полный `param_len` =
/// `d_model · (10 · d_head + 2)`.
pub struct PerFeatureAttention {
    /// Длина временной последовательности (число токенов).
    pub seq_len: usize,
    /// Число входных признаков = число выходных = число голов.
    pub d_model: usize,
    /// Размерность внутреннего пространства одной головы.
    pub d_head: usize,
}

impl PerFeatureAttention {
    /// Значение `d_head` по умолчанию.
    pub const DEFAULT_D_HEAD: usize = 2;

    /// Создаёт слой с `d_head = DEFAULT_D_HEAD`.
    pub fn new(seq_len: usize, d_model: usize) -> Self {
        Self::with_d_head(seq_len, d_model, Self::DEFAULT_D_HEAD)
    }

    /// Создаёт слой с явно заданным `d_head`.
    ///
    /// # Паника
    /// Паникует, если любой из размеров равен нулю.
    pub fn with_d_head(seq_len: usize, d_model: usize, d_head: usize) -> Self {
        assert!(
            seq_len > 0,
            "PerFeatureAttention: seq_len must be positive"
        );
        assert!(
            d_model > 0,
            "PerFeatureAttention: d_model must be positive"
        );
        assert!(
            d_head > 0,
            "PerFeatureAttention: d_head must be positive"
        );
        Self {
            seq_len,
            d_model,
            d_head,
        }
    }

    /// Число параметров одной головы.
    #[inline]
    pub(crate) fn head_param_count(&self) -> usize {
        10 * self.d_head + 2
    }
}

impl UniversalLayer for PerFeatureAttention {
    fn as_per_feature_attention(&self) -> Option<&PerFeatureAttention> {
        Some(self)
    }

    fn param_len(&self) -> usize {
        self.d_model * self.head_param_count()
    }

    fn input_features(&self) -> usize {
        self.seq_len * self.d_model
    }

    fn output_features(&self) -> usize {
        self.seq_len * self.d_model
    }
}