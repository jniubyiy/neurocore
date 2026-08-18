// src/layers/combiner/combiner.rs

pub struct Combiner {
    input_dim: usize,
    output_dim: usize,
}

impl Combiner {
    pub fn new(input_dims: Vec<usize>, output_dim: usize) -> Self {
        assert_eq!(input_dims.len(), 2, "Combiner requires exactly two inputs");
        assert!(input_dims[0] == input_dims[1], "Combiner inputs must have same size for now");
        Self { input_dim: input_dims[0], output_dim }
    }

    pub fn input_dim(&self) -> usize { self.input_dim }
    pub fn output_dim(&self) -> usize { self.output_dim }

    pub fn param_len(&self) -> usize {
        2 * self.output_dim * self.input_dim + self.output_dim
    }
}