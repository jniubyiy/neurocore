use faer::Mat;
use faer::zip;
use crate::neuron::base::Neuron;

pub struct Sigmoid;

impl Neuron for Sigmoid {
    fn apply(&self, x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }

    fn forward_mat(&self, input: &Mat<f32>) -> Mat<f32> {
        zip!(input.as_ref()).map(|x| 1.0 / (1.0 + (-x.0).exp()))
    }
}




