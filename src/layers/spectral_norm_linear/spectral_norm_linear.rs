// src/layers/spectral_norm_linear/spectral_norm_linear.rs

use crate::layers::UniversalLayer;

/// Линейный слой со спектральной нормализацией весов.
///
/// Поддерживает обучаемый масштаб `scale`. Внутри использует степенной
/// метод (power iteration) для оценки спектральной нормы весовой матрицы.
///
/// Формула:
///   sigma = u^T W v
///   W_sn  = W * (scale / sigma)
///   y     = W_sn x + b
///
/// Параметры:
/// - weight: матрица (out_features × in_features)
/// - bias: вектор (out_features)
/// - scale: скаляр
/// Общее число параметров = out_features * in_features + out_features + 1.
///
/// # Состояние forward
///
/// Кэш степенного метода (`u`, `v`, `sigma`) **не хранится** в структуре
/// слоя. Он создаётся per-chunk в `forward_buffered` из `TempMatrixPool`
/// и передаётся в backward через
/// `BufferedContext::SpectralNormLinear { input, u_state, v_state, sigma_state }`.
///
/// Это делает слой безопасным для чанкового распараллеливания: каждый
/// чанк получает свои изолированные векторы `u`/`v` и скаляр `sigma`,
/// гонки за общее состояние слоя не возникает.
///
/// # Замечание о семантике
///
/// В прежней версии векторы `u` и `v` **накапливались** между forward-вызовами
/// (это классический подход power iteration в Spectral Normalization: между
/// шагами SGD векторы сохраняются, чтобы следующий шаг power iteration
/// стартовал близко к текущему приближению). При переносе в
/// `BufferedContext` это свойство теряется: на каждом forward `u` и `v`
/// инициализируются заново (единицами).
///
/// Для корректного приближения `sigma` при одном forward этого достаточно
/// (power iteration сходится за несколько итераций даже из константного
/// старта). Если требуется **точная** эмуляция исходной динамики с
/// персистентными `u`/`v` между эпохами — эту пару нужно вынести в
/// отдельное персистентное поле слоя (аналогично `BatchRenorm1d::state`).
pub struct SpectrallyNormalizedLinear {
    /// Размерность входа.
    pub in_features: usize,
    /// Размерность выхода.
    pub out_features: usize,
}

impl SpectrallyNormalizedLinear {
    /// Создаёт новый слой.
    ///
    /// # Паника
    /// Паникует, если `in_features == 0` или `out_features == 0`.
    pub fn new(in_features: usize, out_features: usize) -> Self {
        assert!(
            in_features > 0 && out_features > 0,
            "SpectrallyNormalizedLinear: in_features and out_features must be positive"
        );
        Self {
            in_features,
            out_features,
        }
    }
}

impl UniversalLayer for SpectrallyNormalizedLinear {
    fn as_spectral_norm_linear(&self) -> Option<&SpectrallyNormalizedLinear> {
        Some(self)
    }

    fn param_len(&self) -> usize {
        self.in_features * self.out_features + self.out_features + 1
    }

    fn input_features(&self) -> usize {
        self.in_features
    }

    fn output_features(&self) -> usize {
        self.out_features
    }
}