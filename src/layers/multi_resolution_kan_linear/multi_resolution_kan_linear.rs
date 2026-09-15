// src/layers/multi_resolution_kan_linear/multi_resolution_kan_linear.rs

use crate::layers::UniversalLayer;

/// Адаптивный KAN-слой с multi-resolution spline (NeuroCore v2).
///
/// Каждый edge (i, j) вычисляется как взвешенная комбинация двух
/// B-spline сеток разного разрешения и SiLU-ветки:
///
///     f_ij(x) = w_c · spline_coarse_ij(x)
///             + w_f · spline_fine_ij(x)
///             + base_weight_ij · silu(x)
///
/// где
///     (w_c, w_f) = softmax([logit_c, logit_f] / T_ij)
///     T_ij = TEMP_MIN + exp(temp_raw_ij).
///
/// Выход слоя:
///     y_j(x_1, ..., x_in) = bias_j + Σ_i f_ij(x_i)
///
/// Отличия от v1:
///   * B-spline порядка k=3 (C²-гладкий) вместо кусочно-линейной интерполяции;
///   * нет жёсткого clamp: за пределами [-1, 1] базисы плавно затухают
///     до нуля, производные остаются непрерывными;
///   * две сетки — coarse (G=3) и fine (G=8) — смешиваются через softmax
///     с обучаемой температурой (exp-параметризация, как в LearnableSoftplus);
///   * SiLU-ветка как в оригинальном KAN (Liu et al. 2024).
pub struct MultiResolutionKANLinear {
    pub in_features: usize,
    pub out_features: usize,
}

impl MultiResolutionKANLinear {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        assert!(
            in_features > 0,
            "MultiResolutionKANLinear: in_features must be positive"
        );
        assert!(
            out_features > 0,
            "MultiResolutionKANLinear: out_features must be positive"
        );
        Self { in_features, out_features }
    }
}

impl UniversalLayer for MultiResolutionKANLinear {
    fn as_multi_resolution_kan_linear(&self) -> Option<&MultiResolutionKANLinear> {
        Some(self)
    }

    fn param_len(&self) -> usize {
        // Раскладка (см. cpu/mod.rs::Offsets):
        //   bias[out]
        // + mix_logits[in·out·2]
        // + mix_temp_raw[in·out]
        // + spline_coarse[in·out·(G_c+k)]
        // + spline_fine[in·out·(G_f+k)]
        // + base_weight[in·out]
        //
        // G_c=3, G_f=8, k=3  ⇒  (G_c+k)=6, (G_f+k)=11
        // total = out + in·out·(2 + 1 + 6 + 11 + 1) = out + in·out·21
        const G_C: usize = 3;
        const G_F: usize = 8;
        const K: usize = 3;
        self.out_features
            + self.in_features * self.out_features * (4 + (G_C + K) + (G_F + K))
    }

    fn input_features(&self) -> usize {
        self.in_features
    }

    fn output_features(&self) -> usize {
        self.out_features
    }
}