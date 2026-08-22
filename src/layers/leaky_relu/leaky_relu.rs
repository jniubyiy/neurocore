// src/layers/leaky_relu/leaky_relu.rs

use crate::layers::UniversalLayer;

pub struct LeakyReLU {
    pub alpha: f32,
}

impl LeakyReLU {
    pub fn new(alpha: f32) -> Self {
        Self { alpha }
    }
}

impl UniversalLayer for LeakyReLU {
    fn as_leaky_relu(&self) -> Option<&LeakyReLU> {
        Some(self)
    }
}