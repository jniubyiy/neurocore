// src/layers/spectral_norm_linear/spectral_norm_linear.rs

use std::sync::Mutex;
use crate::layers::UniversalLayer;

/// Линейный слой со спектральной нормализацией весов.
///
/// Поддерживает обучаемый масштаб `scale`.
/// Внутри хранит векторы `u` и `v` для степенного метода (не обучаются),
/// обновляемые при каждом прямом проходе.
///
/// Формула:
///   sigma = u^T W v
///   W_sn = W * (scale / sigma)
///   y = W_sn x + b
///
/// Параметры:
/// - weight: матрица (out_features × in_features)
/// - bias: вектор (out_features)
/// - scale: скаляр
/// Общее число параметров = out_features * in_features + out_features + 1.
pub struct SpectrallyNormalizedLinear {
    pub in_features: usize,
    pub out_features: usize,
    pub(crate) state: Mutex<SpectralNormState>,
}

/// Внутреннее состояние спектральной нормализации.
pub(crate) struct SpectralNormState {
    /// Вектор u размером in_features.
    pub(crate) u: Vec<f32>,
    /// Вектор v размером out_features.
    pub(crate) v: Vec<f32>,
    /// Флаг инициализации векторов.
    pub(crate) initialized: bool,
    /// Последнее вычисленное значение sigma (для обратного прохода).
    pub(crate) last_sigma: f32,
}

impl SpectrallyNormalizedLinear {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        assert!(in_features > 0 && out_features > 0);
        Self {
            in_features,
            out_features,
            state: Mutex::new(SpectralNormState {
                u: vec![0.0; in_features],
                v: vec![0.0; out_features],
                initialized: false,
                last_sigma: 1.0,
            }),
        }
    }

    /// Возвращает сохранённое значение sigma, вычисленное при последнем прямом проходе.
    pub(crate) fn get_last_sigma(&self) -> f32 {
        self.state.lock().unwrap().last_sigma
    }

    /// Сохраняет новое значение sigma.
    pub(crate) fn set_last_sigma(&self, sigma: f32) {
        self.state.lock().unwrap().last_sigma = sigma;
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