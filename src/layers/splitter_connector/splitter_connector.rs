// src/layers/splitter_connector/splitter_connector.rs

use crate::layers::UniversalLayer;

pub struct SplitterConnector {
    pub dim_a: usize,
    pub dim_b: usize,
}

impl SplitterConnector {
    pub fn new(dim_a: usize, dim_b: usize) -> Self {
        Self { dim_a, dim_b }
    }
}

impl UniversalLayer for SplitterConnector {}