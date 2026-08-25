// src/losses/square/mod.rs

use std::any::Any;
use crate::losses::ElemCube;

/// Квадрат значения: применяется поэлементно.
pub struct Square;

impl ElemCube for Square {
    fn in_features(&self) -> usize { 1 }
    fn out_features(&self) -> usize { 1 }
    fn as_any(&self) -> &dyn Any { self }
}