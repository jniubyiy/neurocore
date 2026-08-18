// src/layers/splitter/splitter.rs

pub struct Splitter {
    input_dim: usize,
    output_dims: Vec<usize>,
}

impl Splitter {
    pub fn new(input_dim: usize, output_dims: Vec<usize>) -> Self {
        assert_eq!(output_dims.len(), 2, "Splitter requires exactly two outputs");
        Self { input_dim, output_dims }
    }

    pub fn input_dim(&self) -> usize { self.input_dim }
    pub fn output_dims(&self) -> &[usize] { &self.output_dims }

    pub fn param_len(&self) -> usize {
        let p = self.output_dims[0];
        let q = self.output_dims[1];
        self.input_dim * p + self.input_dim * q + p + q
    }
}