// src/neuron/types/linear.rs
use faer::Mat;
use crate::neuron::base::Neuron;

/// Одиночный нейрон линейного слоя.
/// Принимает (batch, in_features), возвращает (batch, 1).
pub struct Linear {
    pub weights: Vec<f32>,  // длина in_features
    pub bias: f32,
}

impl Linear {
    pub fn new(weights: Vec<f32>, bias: f32) -> Self {
        Self { weights, bias }
    }
}

impl Neuron for Linear {
    fn forward_mat(&self, input: &Mat<f32>) -> Mat<f32> {
        let batch = input.nrows();
        let in_features = input.ncols();
        assert_eq!(self.weights.len(), in_features,
            "Linear neuron: weights length must match input columns");

        let mut output = Mat::zeros(batch, 1);
        for i in 0..batch {
            let mut sum = self.bias;
            for j in 0..in_features {
                sum += input[(i, j)] * self.weights[j];
            }
            output[(i, 0)] = sum;
        }
        output
    }
}



