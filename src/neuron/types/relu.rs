use crate::neuron::base::Neuron;

pub struct ReLU;

impl Neuron for ReLU {
    fn apply(&self, x: f32) -> f32 {
        if x > 0.0 { x } else { 0.0 }
    }
}




