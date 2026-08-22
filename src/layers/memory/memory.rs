// src/layers/memory/memory.rs

use std::sync::Mutex;

use crate::layers::UniversalLayer;

pub struct Memory {
    pub(crate) features: usize,
    pub alpha: f32,
    pub(crate) cells: Mutex<Vec<Option<f32>>>, // None означает, что якорь ещё не инициализирован
}

impl Memory {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        assert_eq!(in_features, out_features,
            "Memory: in_features must equal out_features");
        let mut cells = Vec::with_capacity(2 * in_features);
        cells.resize(2 * in_features, None);
        Self {
            features: in_features,
            alpha: 0.1,
            cells: Mutex::new(cells),
        }
    }
}

impl UniversalLayer for Memory {
    fn as_memory(&self) -> Option<&Memory> {
        Some(self)
    }

    fn param_len(&self) -> usize {
        0
    }

    fn input_features(&self) -> usize {
        self.features
    }

    fn output_features(&self) -> usize {
        self.features
    }
}