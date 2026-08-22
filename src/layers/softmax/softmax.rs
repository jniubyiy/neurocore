// src/layers/softmax/softmax.rs

use crate::layers::UniversalLayer;

pub struct Softmax;

impl Softmax {
    pub fn new() -> Self {
        Self
    }
}

impl UniversalLayer for Softmax {
    fn as_softmax(&self) -> Option<&Softmax> {
        Some(self)
    }
}