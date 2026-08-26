// src/losses/abs_diff/mod.rs

use std::any::Any;
use crate::losses::ElemCube;

/// Абсолютная разность двух входов (pred и target).
#[derive(Debug)]
pub struct AbsDiff {
    features: usize,
}

impl AbsDiff {
    pub fn new(pred_features: usize) -> Self {
        assert!(pred_features > 0, "AbsDiff: pred_features must be positive");
        Self { features: pred_features }
    }
}

impl Default for AbsDiff {
    fn default() -> Self {
        Self { features: 1 }
    }
}

impl ElemCube for AbsDiff {
    fn in_features(&self) -> usize {
        2 * self.features
    }

    fn out_features(&self) -> usize {
        self.features
    }

    fn as_any(&self) -> &dyn Any { self }
}

mod cpu;
pub mod gpu;