// src/neuron/types/sigmoid.rs

use faer::Mat;
use crate::neuron::base::Neuron;

pub struct Sigmoid;

impl Neuron for Sigmoid {
    fn forward_mat(&self, input: &Mat<f32>) -> Mat<f32> {
        let mut out = Mat::zeros(input.nrows(), 1);
        for i in 0..input.nrows() {
            out[(i, 0)] = 1.0 / (1.0 + (-input[(i, 0)]).exp());
        }
        out
    }
}




