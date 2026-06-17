use crate::tensor::Tensor1D;
use crate::neuron::base::Neuron;

pub struct Linear {
    pub weights: Vec<f32>,
    pub bias: f32,
}

impl Linear {
    pub fn new(weights: Vec<f32>, bias: f32) -> Self {
        Linear { weights, bias }
    }

    /// Статический метод для расчёта одного нейрона без владения данными.
    pub fn forward_slice(weights: &[f32], bias: f32, input: &[f32]) -> f32 {
        assert_eq!(weights.len(), input.len(), "Linear forward_slice: weight/input length mismatch");
        let mut sum = bias;
        for i in 0..weights.len() {
            sum += weights[i] * input[i];
        }
        sum
    }
}

impl Neuron for Linear {
    fn apply(&self, _x: f32) -> f32 {
        panic!("Linear neuron does not support element‑wise apply; use forward()");
    }

    fn forward(&self, input: &Tensor1D) -> Tensor1D {
        assert_eq!(input.len(), self.weights.len(), "Linear: input length must match weights length");
        let sum = self.bias + self.weights.iter().zip(&input.data).map(|(w, x)| w * x).sum::<f32>();
        Tensor1D::from_scalar(sum)
    }
}





