// src/layers/soft_sparse_gate/soft_sparse_gate.rs

use crate::layers::UniversalLayer;

pub struct SoftSparseGate {
    pub in_features: usize,
    pub temperature: f32,
}

impl SoftSparseGate {
    pub fn new(in_features: usize, temperature: f32) -> Self {
        assert!(temperature > 0.0, "SoftSparseGate: temperature must be positive");
        Self { in_features, temperature }
    }
}

impl UniversalLayer for SoftSparseGate {
    fn as_soft_sparse_gate(&self) -> Option<&SoftSparseGate> {
        Some(self)
    }

    fn param_len(&self) -> usize {
        self.in_features
    }

    fn input_features(&self) -> usize {
        self.in_features
    }

    fn output_features(&self) -> usize {
        self.in_features
    }
}