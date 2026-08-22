// src/layers/linear/linear.rs

use crate::layers::UniversalLayer;

pub struct Linear {
    pub(crate) in_features: usize,
    pub(crate) out_features: usize,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        Self { in_features, out_features }
    }
}

impl UniversalLayer for Linear {
    fn as_linear(&self) -> Option<&Linear> {
        Some(self)
    }

    fn param_len(&self) -> usize {
        self.in_features * self.out_features + self.out_features
    }

    fn input_features(&self) -> usize {
        self.in_features
    }

    fn output_features(&self) -> usize {
        self.out_features
    }
}