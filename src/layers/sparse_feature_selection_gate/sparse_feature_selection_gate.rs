// src/layers/sparse_feature_selection_gate/sparse_feature_selection_gate.rs

use crate::layers::UniversalLayer;

/// Слой SparseFeatureSelectionGate.
///
/// Выполняет обучаемое поэлементное стробирование признаков.
/// Маска вычисляется как sigmoid(logits / temperature).
/// Параметры:
/// - logits: вектор длины features (обучаемые логиты важности признаков)
/// - temperature: скаляр (обучаемый параметр, управляет крутизной сигмоиды)
/// Общее число параметров = features + 1.
pub struct SparseFeatureSelectionGate {
    pub features: usize,
}

impl SparseFeatureSelectionGate {
    pub fn new(features: usize) -> Self {
        assert!(features > 0, "SparseFeatureSelectionGate: features must be positive");
        Self { features }
    }
}

impl UniversalLayer for SparseFeatureSelectionGate {
    fn as_sparse_feature_selection_gate(&self) -> Option<&SparseFeatureSelectionGate> {
        Some(self)
    }

    fn param_len(&self) -> usize {
        self.features + 1
    }

    fn input_features(&self) -> usize {
        self.features
    }

    fn output_features(&self) -> usize {
        self.features
    }
}