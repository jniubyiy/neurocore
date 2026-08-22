// src/layers/dual_anchor/dual_anchor.rs

use crate::layers::UniversalLayer;

pub struct DualAnchor {
    pub features: usize,
}

impl DualAnchor {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        assert_eq!(in_features, out_features,
            "DualAnchor: in_features must equal out_features");
        Self { features: in_features }
    }
}

impl UniversalLayer for DualAnchor {
    fn as_dual_anchor(&self) -> Option<&DualAnchor> {
        Some(self)
    }

    fn param_len(&self) -> usize {
        2 * self.features + 1
    }

    fn input_features(&self) -> usize {
        self.features
    }

    fn output_features(&self) -> usize {
        self.features
    }
}