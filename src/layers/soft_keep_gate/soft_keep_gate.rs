// src/layers/soft_keep_gate/soft_keep_gate.rs

use crate::layers::UniversalLayer;

pub struct SoftKeepGate {
    pub in_features: usize,
    pub temperature: f32,
}

impl SoftKeepGate {
    pub fn new(in_features: usize, temperature: f32) -> Self {
        assert!(temperature > 0.0, "SoftKeepGate: temperature must be positive");
        Self { in_features, temperature }
    }
}

impl UniversalLayer for SoftKeepGate {
    fn as_soft_keep_gate(&self) -> Option<&SoftKeepGate> {
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