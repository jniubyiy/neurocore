use crate::tensor::Tensor1D;
use crate::jacobian::Jacobian;
use crate::neuron::base::Neuron;

pub struct Identity;

impl Neuron for Identity {
    fn forward(&self, input: &Tensor1D, j_input: &Jacobian) -> (Tensor1D, Jacobian) {
        (input.clone(), j_input.clone())
    }
}