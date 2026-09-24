// src/layers/adaptive_space_compress/adaptive_space_compress.rs

use crate::layers::UniversalLayer;

/// Слой AdaptiveSpaceCompress — обучаемое сжатие пространства признаков
/// через мягкую группировку в плоскости.
///
/// # Ключевое свойство
///
/// `param_len` **не зависит** от `in_features`. Это даёт:
///
///   * единую аллокацию параметров в `ParamStore` для любых входных
///     размеров;
///   * возможность использовать один и тот же слой на входах разной
///     длины;
///   * assign вычисляется из **позиции признака** динамически, а не
///     хранится таблицей `in_features × p_max`.
///
/// Фактический `in_features` берётся из `input.cols()` в момент forward
/// и сохраняется в контексте — backward использует тот же размер.
///
/// # Формула
///
///   pos_j            = (j + 0.5) / in_features       ∈ (0, 1)
///   L[j, p]          = -(pos_j - center[p])² + b_L[p]
///   assign[j, p]     = softmax_p(L[j, p])
///   value[r, p]      = Σ_j assign[j, p] · x[r, j]
///   compressed[r, p] = value[r, p] · compress[p] · w_p
///   y[r, k]          = b[k] + Σ_p compressed[r, p] · W[p, k]
///
/// где w_p — вес плоскости:
///
///   p_soft(p_raw) = MIN_PLANES
///                 + Σ_{k=1}^{p_max−MIN_PLANES} σ((p_raw − θ_k)/τ),
///   θ_k = k − 1,  MIN_PLANES = 1,  τ = 0.5
///
///   w_p(p_soft) = 0.5·(1 + tanh(K·((p_soft − p) − 0.5))),  K = 4.0
///
/// Аппарат мягкого числа активных плоскостей заимствован из
/// `LinearAttention` (`compute_h_soft` / `head_weight`).
///
/// # Раскладка параметров
///
/// Порядок в общем буфере (смещение `slice.start`):
///
/// | Смещение          | Размер           | Что                          |
/// |-------------------|------------------|------------------------------|
/// | `0`               | `p_max`          | `center[p]`                  |
/// | `p_max`           | `p_max`          | `b_L[p]`                     |
/// | `2·p_max`         | `p_max`          | `compress[p]`                |
/// | `3·p_max`         | `p_max · out`    | `W[p, k]` (row-major по p)   |
/// | `+p_max·out`      | `out`            | `b[k]`                       |
/// | `+out`            | `1`              | `p_raw`                      |
///
/// Итого: `p_max · (3 + out_features) + out_features + 1`.
///
/// Каноническая инициализация (см. `bridge_v2.rs`):
///   * `center[p] = p / (p_max − 1)` (равномерно по позициям; при
///     `p_max == 1` — `0.5`);
///   * `b_L[p] = 0.0`;
///   * `compress[p] = 1.0`.
pub struct AdaptiveSpaceCompress {
    /// Размерность выхода. Жёстко задана.
    pub out_features: usize,
    /// Максимальное число плоскостей. Верхняя граница для `p_soft`.
    pub p_max: usize,
}

impl AdaptiveSpaceCompress {
    /// Создаёт слой.
    ///
    /// # Паника
    /// Паникует, если `out_features == 0` или `p_max == 0`.
    pub fn new(out_features: usize, p_max: usize) -> Self {
        assert!(
            out_features > 0,
            "AdaptiveSpaceCompress: out_features must be positive"
        );
        assert!(p_max > 0, "AdaptiveSpaceCompress: p_max must be positive");
        Self {
            out_features,
            p_max,
        }
    }

    /// Смещение блока `center`.
    #[inline]
    pub(crate) fn center_offset(&self) -> usize {
        0
    }

    /// Смещение блока `b_L`.
    #[inline]
    pub(crate) fn b_l_offset(&self) -> usize {
        self.p_max
    }

    /// Смещение блока `compress`.
    #[inline]
    pub(crate) fn compress_offset(&self) -> usize {
        2 * self.p_max
    }

    /// Смещение блока `W`.
    #[inline]
    pub(crate) fn w_offset(&self) -> usize {
        3 * self.p_max
    }

    /// Смещение блока `b`.
    #[inline]
    pub(crate) fn b_offset(&self) -> usize {
        self.w_offset() + self.p_max * self.out_features
    }

    /// Смещение `p_raw`.
    #[inline]
    pub(crate) fn p_raw_offset(&self) -> usize {
        self.b_offset() + self.out_features
    }
}

impl UniversalLayer for AdaptiveSpaceCompress {
    fn as_adaptive_space_compress(&self) -> Option<&AdaptiveSpaceCompress> {
        Some(self)
    }

    /// `param_len` **не зависит** от `in_features`.
    fn param_len(&self) -> usize {
        self.p_max * (3 + self.out_features) + self.out_features + 1
    }

    /// Возвращает 0 — сигнал внешним потребителям, что реальный размер
    /// входа определяется в forward (`input.cols()`) и передаётся через
    /// контекст. Универсальный фолбэк (`input_features_for`) не должен
    /// использоваться для этого слоя: в backward мы явно берём размер
    /// из `BufferedContext::AdaptiveSpaceCompress { input, .. }`.
    fn input_features(&self) -> usize {
        0
    }

    fn output_features(&self) -> usize {
        self.out_features
    }
}

