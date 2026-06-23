// src/neuron/base.rs
use faer::Mat;

pub trait Neuron {
    fn forward_mat(&self, input: &Mat<f32>) -> Mat<f32>;
}


