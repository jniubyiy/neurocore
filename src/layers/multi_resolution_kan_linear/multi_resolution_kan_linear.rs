// src/layers/multi_resolution_kan_linear/multi_resolution_kan_linear.rs

use crate::layers::UniversalLayer;

/// Слой MultiResolutionKANLinear — упрощённая версия KAN с двумя разрешениями.
///
/// Для каждой пары (входной признак i, выходной нейрон j) хранятся коэффициенты
/// для двух сеток: грубой (coarse) размера 4 и точной (fine) размера 8.
/// Используется линейная интерполяция входного значения по равномерной сетке на [-1, 1].
///
/// Параметры:
/// - коэффициенты coarse: in_features * out_features * 4
/// - коэффициенты fine:   in_features * out_features * 8
/// - bias: out_features
/// Общее число параметров = in_features * out_features * (4 + 8) + out_features.
pub struct MultiResolutionKANLinear {
    pub in_features: usize,
    pub out_features: usize,
}

impl MultiResolutionKANLinear {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        assert!(in_features > 0 && out_features > 0);
        Self { in_features, out_features }
    }
}

impl UniversalLayer for MultiResolutionKANLinear {
    fn as_multi_resolution_kan_linear(&self) -> Option<&MultiResolutionKANLinear> {
        Some(self)
    }

    fn param_len(&self) -> usize {
        let coarse = 4;
        let fine = 8;
        self.in_features * self.out_features * (coarse + fine) + self.out_features
    }

    fn input_features(&self) -> usize {
        self.in_features
    }

    fn output_features(&self) -> usize {
        self.out_features
    }
}