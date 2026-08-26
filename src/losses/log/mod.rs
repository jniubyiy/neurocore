// src/losses/log/mod.rs

use std::any::Any;
use crate::losses::ElemCube;

/// Натуральный логарифм (поэлементно).
#[derive(Debug)]
pub struct Log;

impl ElemCube for Log {
    fn in_features(&self) -> usize { 1 }
    fn out_features(&self) -> usize { 1 }
    fn as_any(&self) -> &dyn Any { self }
}

mod cpu;
pub mod gpu;