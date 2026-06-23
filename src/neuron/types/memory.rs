// src/neuron/types/memory.rs

use faer::Mat;
use crate::neuron::base::Neuron;

/// Одиночный нейрон Memory: два обучаемых вектора весов и температура.
/// Принимает (batch, in_features), возвращает (batch, 1).
pub struct Memory {
    pub weight0: Vec<f32>,   // длина in_features
    pub weight1: Vec<f32>,   // длина in_features
    pub temperature: f32,    // > 0
}

impl Memory {
    pub fn new(weight0: Vec<f32>, weight1: Vec<f32>, temperature: f32) -> Self {
        assert_eq!(weight0.len(), weight1.len());
        assert!(temperature > 0.0);
        Self { weight0, weight1, temperature }
    }
}

impl Neuron for Memory {
    fn forward_mat(&self, input: &Mat<f32>) -> Mat<f32> {
        let batch = input.nrows();
        let in_features = input.ncols();
        assert_eq!(self.weight0.len(), in_features, "Memory neuron: weight length must match input columns");

        let mut output = Mat::zeros(batch, 1);
        for i in 0..batch {
            let mut dot0 = 0.0;
            let mut dot1 = 0.0;
            for j in 0..in_features {
                dot0 += input[(i, j)] * self.weight0[j];
                dot1 += input[(i, j)] * self.weight1[j];
            }
            let scaled0 = dot0 / self.temperature;
            let scaled1 = dot1 / self.temperature;
            let max_logit = scaled0.max(scaled1);
            let exp0 = (scaled0 - max_logit).exp();
            let exp1 = (scaled1 - max_logit).exp();
            let sum = exp0 + exp1;
            let soft0 = exp0 / sum;
            let soft1 = exp1 / sum;
            output[(i, 0)] = soft0 * dot0 + soft1 * dot1;
        }
        output
    }
}