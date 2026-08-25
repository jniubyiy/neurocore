// src/losses/neg/mod.rs

use std::any::Any;
use crate::losses::ElemCube;

/// Унарный минус (поэлементно).
pub struct Neg;

impl ElemCube for Neg {
    fn in_features(&self) -> usize { 1 }
    fn out_features(&self) -> usize { 1 }
    fn as_any(&self) -> &dyn Any { self }
}
