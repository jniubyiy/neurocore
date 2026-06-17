use crate::neuron::base::Neuron;

pub struct Identity;

impl Neuron for Identity {
    fn apply(&self, x: f32) -> f32 { x }
}