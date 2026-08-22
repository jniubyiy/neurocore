// src/layers/sigmoid/sigmoid.rs

use crate::layers::UniversalLayer;

pub struct Sigmoid;

impl Sigmoid {
    pub fn new() -> Self {
        Self
    }
}

impl UniversalLayer for Sigmoid {
    fn as_sigmoid(&self) -> Option<&Sigmoid> {
        Some(self)
    }
}