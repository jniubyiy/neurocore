// src/neuron/types/soft_sparse_gate.rs
use faer::Mat;
use crate::neuron::base::Neuron;

pub struct SoftSparseGate {
    pub threshold: f32,
    pub temperature: f32,
}

impl SoftSparseGate {
    pub fn new(threshold: f32, temperature: f32) -> Self {
        assert!(temperature > 0.0);
        Self { threshold, temperature }
    }
}

impl Neuron for SoftSparseGate {
    fn forward_mat(&self, input: &Mat<f32>) -> Mat<f32> {
        let mut out = Mat::zeros(input.nrows(), input.ncols());
        for i in 0..input.nrows() {
            for j in 0..input.ncols() {
                let x = input[(i, j)];
                let abs_x = x.abs();
                let z = (abs_x - self.threshold) / self.temperature;
                let gate = 1.0 / (1.0 + (-z).exp());
                out[(i, j)] = x * gate;
            }
        }
        out
    }
}





