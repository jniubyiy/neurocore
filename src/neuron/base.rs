use crate::tensor::Tensor1D;
use crate::jacobian::Jacobian;

/// Нейрон с одним входом.
pub trait Neuron {
    fn forward(&self, input: &Tensor1D, j_input: &Jacobian) -> (Tensor1D, Jacobian);
}





