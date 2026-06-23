// src/neuron/types/soft_keep_gate.rs
use faer::Mat;
use crate::neuron::base::Neuron;

pub struct SoftKeepGate {
    pub threshold: f32,
    pub temperature: f32,
}

impl SoftKeepGate {
    pub fn new(threshold: f32, temperature: f32) -> Self {
        assert!(temperature > 0.0);
        Self { threshold, temperature }
    }
}

impl Neuron for SoftKeepGate {
    fn forward_mat(&self, input: &Mat<f32>) -> Mat<f32> {
        let mut out = Mat::zeros(input.nrows(), input.ncols());
        for i in 0..input.nrows() {
            for j in 0..input.ncols() {
                let x = input[(i, j)];
                let abs_x = x.abs();
                let z = (self.threshold - abs_x) / self.temperature;
                let gate = 1.0 / (1.0 + (-z).exp());
                out[(i, j)] = x * gate;
            }
        }
        out
    }
}



