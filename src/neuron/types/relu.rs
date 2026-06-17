use faer::Mat;
use faer::zip;
use crate::neuron::base::Neuron;

pub struct ReLU;

impl Neuron for ReLU {
    fn apply(&self, x: f32) -> f32 { x.max(0.0) }

    fn forward_mat(&self, input: &Mat<f32>) -> Mat<f32> {
        zip!(input.as_ref()).map(|x| x.0.max(0.0))
    }
}




