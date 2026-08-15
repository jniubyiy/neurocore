// src/layers/layers_special/expand_dim.rs

use crate::layers::UniversalLayer;

pub struct Unsqueeze {
    pub target_dims: Vec<usize>,
}

impl Unsqueeze {
    pub fn with_target_dims(target_dims: Vec<usize>) -> Self {
        Self { target_dims }
    }
}

impl UniversalLayer for Unsqueeze {
    fn as_unsqueeze(&self) -> Option<&Unsqueeze> {
        Some(self)
    }
}





