// src/losses/mul/mod.rs

use std::any::Any;
use crate::losses::ElemCube;

/// Поэлементное умножение двух входов (pred и target).
#[derive(Debug)]
pub struct Mul {
    features: usize,
}

impl Mul {
    pub fn new(pred_features: usize) -> Self {
        assert!(pred_features > 0, "Mul: pred_features must be positive");
        Self { features: pred_features }
    }
}

impl Default for Mul {
    fn default() -> Self {
        Self { features: 1 }
    }
}

impl ElemCube for Mul {
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