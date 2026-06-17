use crate::tensor::Tensor1D;
use crate::neuron::base::Neuron;

pub struct Softmax;

impl Neuron for Softmax {
    fn apply(&self, _x: f32) -> f32 {
        panic!("Softmax cannot be applied element‑wise; use forward() for a full vector.");
    }

    fn forward(&self, input: &Tensor1D) -> Tensor1D {
        let max_val = input.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = input.data.iter().map(|&x| (x - max_val).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let out: Vec<f32> = exps.iter().map(|&e| e / sum).collect();
        Tensor1D::new(out)
    }
}




