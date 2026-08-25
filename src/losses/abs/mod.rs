// src/losses/abs/mod.rs

use std::any::Any;
use crate::losses::ElemCube;

/// Абсолютное значение (поэлементно).
pub struct Abs;

impl ElemCube for Abs {
    fn in_features(&self) -> usize { 1 }
    fn out_features(&self) -> usize { 1 }
    fn as_any(&self) -> &dyn Any { self }
}
