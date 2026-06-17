use crate::neuron::base::Neuron;

pub struct Tanh;

impl Neuron for Tanh {
    fn apply(&self, x: f32) -> f32 {
        x.tanh()
    }
}




