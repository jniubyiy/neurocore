// src/losses/sub/mod.rs

use std::any::Any;
use crate::losses::ElemCube;

#[derive(Debug)]
pub struct Sub {
    features: usize,
}

impl Sub {
    pub fn new(pred_features: usize) -> Self {
        assert!(pred_features > 0, "Sub: pred_features must be positive");
        Self { features: pred_features }
    }
}

impl Default for Sub {
    fn default() -> Self {
        Self { features: 1 }
    }
}

impl ElemCube for Sub {
    fn in_features(&self) -> usize { 2 * self.features }
    fn out_features(&self) -> usize { self.features }
    fn as_any(&self) -> &dyn Any { self }
}

mod cpu;
pub mod gpu;   // <-- добавлена эта строка
