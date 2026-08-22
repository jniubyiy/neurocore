// src/layers/relu/relu.rs

use crate::layers::UniversalLayer;

pub struct ReLU;

impl ReLU {
    pub fn new() -> Self {
        Self
    }
}

impl UniversalLayer for ReLU {
    fn as_relu(&self) -> Option<&ReLU> {
        Some(self)
    }
}