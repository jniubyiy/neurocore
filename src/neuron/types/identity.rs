use faer::Mat;
use crate::neuron::base::Neuron;

pub struct Identity;

impl Neuron for Identity {
    fn apply(&self, x: f32) -> f32 { x }

    fn forward_mat(&self, input: &Mat<f32>) -> Mat<f32> {
        input.clone()
    }
}