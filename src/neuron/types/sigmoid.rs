use crate::neuron::base::Neuron;

pub struct Sigmoid;

impl Neuron for Sigmoid {
    fn apply(&self, x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }
}





