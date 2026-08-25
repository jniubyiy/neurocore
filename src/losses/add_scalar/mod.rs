// src/losses/add_scalar/mod.rs

use std::any::Any;
use crate::losses::ElemCube;

/// Добавляет скаляр ко всем элементам (поэлементно).
pub struct AddScalar(pub f32);

impl ElemCube for AddScalar {
    fn in_features(&self) -> usize { 1 }
    fn out_features(&self) -> usize { 1 }
    fn as_any(&self) -> &dyn Any { self }
}