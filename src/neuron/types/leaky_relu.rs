use crate::neuron::base::Neuron;

pub struct LeakyReLU {
    pub alpha: f32,
}

impl LeakyReLU {
    pub fn new(alpha: f32) -> Self { Self { alpha } }
}

impl Neuron for LeakyReLU {
    fn apply(&self, x: f32) -> f32 {
        if x > 0.0 { x } else { self.alpha * x }
    }
}




