// src/losses/log1p/mod.rs

use std::any::Any;
use crate::losses::ElemCube;

/// Натуральный логарифм от (x + 1) (поэлементно).
pub struct Log1p;

impl ElemCube for Log1p {
    fn in_features(&self) -> usize { 1 }
    fn out_features(&self) -> usize { 1 }
    fn as_any(&self) -> &dyn Any { self }
}
