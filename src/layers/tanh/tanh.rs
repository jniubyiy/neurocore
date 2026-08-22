// src/layers/tanh/tanh.rs

use crate::layers::UniversalLayer;

pub struct Tanh;

impl Tanh {
    pub fn new() -> Self {
        Self
    }
}

impl UniversalLayer for Tanh {
    fn as_tanh(&self) -> Option<&Tanh> {
        Some(self)
    }
}