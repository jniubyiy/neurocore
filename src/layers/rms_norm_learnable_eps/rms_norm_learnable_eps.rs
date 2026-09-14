// src/layers/rms_norm_learnable_eps/rms_norm_learnable_eps.rs

use crate::layers::UniversalLayer;

/// Нижняя граница для ε.
///
/// Гарантирует `mean_sq + ε > 0` при любых значениях обучаемого `eps_raw`,
/// что устраняет NaN из `sqrt(mean_sq + ε)` при обучении.
///
/// Используемая параметризация:
///
/// ```text
///     ε(eps_raw) = EPS_MIN + exp(eps_raw)
///     ∂ε/∂eps_raw = exp(eps_raw) = ε − EPS_MIN
/// ```
///
/// При инициализации `eps_raw ≈ 0` получаем `ε ≈ 1`; чтобы получить
/// «стандартный» RMS-эпсилон (например, `1e-5`), нужно инициализировать
/// `eps_raw ≈ ln(1e-5) ≈ −11.51`.
pub const EPS_MIN: f32 = 1e-6;

/// Слой RMSNormWithLearnableEpsilon.
///
/// Выполняет RMS-нормализацию с обучаемым параметром epsilon.
/// Формула: `y = x / sqrt(mean(x^2) + ε) * γ`.
///
/// Параметры слоя (в порядке в общем буфере):
/// - `gamma` (вектор длины `features`);
/// - `eps_raw` (вектор длины `features`) — **сырое** значение, из которого
///   вычисляется `ε = EPS_MIN + exp(eps_raw)`. Хранение именно `eps_raw`
///   (а не самого `ε`) обеспечивает гладкую нелинейную параметризацию,
///   при которой `ε > 0` гарантированно.
///
/// Общее количество параметров = `2 * features`.
pub struct RMSNormWithLearnableEpsilon {
    /// Количество признаков.
    pub features: usize,
}

impl RMSNormWithLearnableEpsilon {
    /// Создаёт новый слой.
    ///
    /// # Паника
    /// Паникует, если `features == 0`.
    pub fn new(features: usize) -> Self {
        assert!(features > 0, "RMSNormWithLearnableEpsilon: features must be positive");
        Self { features }
    }
}

impl UniversalLayer for RMSNormWithLearnableEpsilon {
    fn as_rms_norm_learnable_eps(&self) -> Option<&RMSNormWithLearnableEpsilon> {
        Some(self)
    }

    fn param_len(&self) -> usize {
        2 * self.features
    }

    fn input_features(&self) -> usize {
        self.features
    }

    fn output_features(&self) -> usize {
        self.features
    }
}