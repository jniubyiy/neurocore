// src/neuron/types/leaky_relu.rs
use faer::Mat;
use faer::zip;
use crate::neuron::base::Neuron;

pub struct LeakyReLU {
    pub alpha: f32,
}

impl LeakyReLU {
    pub fn new(alpha: f32) -> Self { Self { alpha } }
}

impl Neuron for LeakyReLU {
    fn forward_mat(&self, input: &Mat<f32>) -> Mat<f32> {
        let alpha = self.alpha;
        zip!(input.as_ref()).map(|x| {
            let val = x.0;
            if *val > 0.0 { *val } else { alpha * (*val) }
        })
    }
}



