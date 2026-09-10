// src/layers/spectral_norm_linear/spectral_norm_linear.rs

use std::sync::RwLock;
use crate::layers::UniversalLayer;

/// Внутреннее состояние спектральной нормализации.
///
/// Содержит векторы `u` (длины `in_features`) и `v` (длины `out_features`),
/// используемые в степенном методе, а также последнее вычисленное значение
/// `sigma` (используется в обратном проходе).
///
/// Поле `initialized` показывает, была ли выполнена инициализация векторов
/// `u` и `v` (по первому прямому проходу). Пока `initialized == false`,
/// содержимое `u` и `v` не определено и не используется.
pub(crate) struct SpectralNormState {
    /// Вектор `u` длины `in_features`.
    pub(crate) u: Vec<f32>,
    /// Вектор `v` длины `out_features`.
    pub(crate) v: Vec<f32>,
    /// Флаг инициализации векторов `u` и `v`.
    pub(crate) initialized: bool,
    /// Последнее вычисленное значение `sigma`.
    pub(crate) last_sigma: f32,
}

/// Линейный слой со спектральной нормализацией весов.
///
/// Поддерживает обучаемый масштаб `scale`.
/// Внутри хранит векторы `u` и `v` для степенного метода (не обучаются),
/// обновляемые при каждом прямом проходе.
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
pub struct SpectrallyNormalizedLinear {
    pub in_features: usize,
    pub out_features: usize,
    /// Состояние слоя.
    pub(crate) state: RwLock<SpectralNormState>,
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
            state: RwLock::new(SpectralNormState {
                u: vec![0.0; in_features],
                v: vec![0.0; out_features],
                initialized: false,
                last_sigma: 1.0,
            }),
        }
    }

    /// Возвращает сохранённое значение `sigma`, вычисленное при последнем
    /// прямом проходе.
    pub(crate) fn get_last_sigma(&self) -> f32 {
        self.state.read().unwrap().last_sigma
    }

    /// Сохраняет новое значение `sigma`.
    pub(crate) fn set_last_sigma(&self, sigma: f32) {
        self.state.write().unwrap().last_sigma = sigma;
    }

    /// Возвращает `true`, если векторы `u` и `v` были инициализированы.
    pub(crate) fn is_initialized(&self) -> bool {
        self.state.read().unwrap().initialized
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