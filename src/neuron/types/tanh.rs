// src/neuron/types/tanh.rs
use faer::Mat;
use faer::zip;
use crate::neuron::base::Neuron;

pub struct Tanh;

impl Neuron for Tanh {
    fn forward_mat(&self, input: &Mat<f32>) -> Mat<f32> {
        zip!(input.as_ref()).map(|x| x.0.tanh())
    }
}


