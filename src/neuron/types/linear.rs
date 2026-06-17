use faer::Mat;
use faer::zip;
use crate::neuron::base::Neuron;

pub struct Linear {
    pub weights: Vec<f32>,
    pub bias: f32,
}

impl Linear {
    pub fn new(weights: Vec<f32>, bias: f32) -> Self {
        Linear { weights, bias }
    }

    pub fn forward_slice(weights: &[f32], bias: f32, input: &[f32]) -> f32 {
        assert_eq!(weights.len(), input.len());
        let mut sum = bias;
        for i in 0..weights.len() {
            sum += weights[i] * input[i];
        }
        sum
    }

    fn weights_col(&self) -> Mat<f32> {
        Mat::from_fn(self.weights.len(), 1, |i, _| self.weights[i])
    }
}

impl Neuron for Linear {
    fn apply(&self, _x: f32) -> f32 {
        panic!("Linear neuron does not support element‑wise apply; use forward()");
    }

    fn forward(&self, input: &crate::tensor::Tensor1D) -> crate::tensor::Tensor1D {
        assert_eq!(input.len(), self.weights.len());
        let sum = self.bias + self.weights.iter().zip(&input.data).map(|(w, x)| w * x).sum::<f32>();
        crate::tensor::Tensor1D::from_scalar(sum)
    }

    fn forward_mat(&self, input: &Mat<f32>) -> Mat<f32> {
        let out = input * &self.weights_col(); // (batch x 1)
        zip!(out.as_ref()).map(|x| x.0 + self.bias)
    }
}





