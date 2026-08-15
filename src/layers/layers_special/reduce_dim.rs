// src/layers/layers_special/reduce_dim.rs

use crate::layers::UniversalLayer;

pub struct ReduceMean {
    pub target_dims: Vec<usize>,
}

impl ReduceMean {
    pub fn with_dims(input_dims: Vec<usize>, target_dims: Vec<usize>) -> Self {
        assert_eq!(input_dims.len(), target_dims.len() + 1,
            "ReduceMean: target_dims must have exactly one less dimension than input_dims");
        let input_total: usize = input_dims.iter().product();
        let target_total: usize = target_dims.iter().product();
        assert_eq!(input_total, target_total,
            "ReduceMean: total number of elements must be conserved");
        Self { target_dims }
    }

    pub fn with_target_dims(target_dims: Vec<usize>) -> Self {
        Self { target_dims }
    }
}

impl UniversalLayer for ReduceMean {
    fn as_reduce_mean(&self) -> Option<&ReduceMean> {
        Some(self)
    }
}





