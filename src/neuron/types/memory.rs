use crate::tensor::Tensor1D;
use crate::neuron::base::Neuron;

pub struct Memory {
    pub memory0: Vec<f32>,
    pub memory1: Vec<f32>,
    pub temperature: f32,
}

impl Memory {
    pub fn new(memory0: Vec<f32>, memory1: Vec<f32>, temperature: f32) -> Self {
        assert_eq!(memory0.len(), memory1.len());
        assert!(temperature > 0.0);
        Memory { memory0, memory1, temperature }
    }

    pub fn param_count(&self) -> usize {
        2 * self.memory0.len() + 1
    }

    pub fn get_params(&self) -> Vec<f32> {
        let mut v = Vec::with_capacity(self.param_count());
        v.extend_from_slice(&self.memory0);
        v.extend_from_slice(&self.memory1);
        v.push(self.temperature);
        v
    }

    pub fn set_params(&mut self, values: &[f32]) {
        let n = self.memory0.len();
        assert_eq!(values.len(), 2 * n + 1);
        self.memory0.copy_from_slice(&values[..n]);
        self.memory1.copy_from_slice(&values[n..2 * n]);
        self.temperature = values[2 * n];
    }
}

impl Neuron for Memory {
    fn apply(&self, _x: f32) -> f32 {
        panic!("Memory does not support element‑wise apply; use forward()");
    }

    fn forward(&self, input: &Tensor1D) -> Tensor1D {
        assert_eq!(input.len(), self.memory0.len());
        let t_val = self.temperature;
        let m0 = &self.memory0;
        let m1 = &self.memory1;

        let mut dot0 = 0.0;
        let mut dot1 = 0.0;
        for i in 0..input.len() {
            dot0 += input.data[i] * m0[i];
            dot1 += input.data[i] * m1[i];
        }

        let logit0 = dot0 / t_val;
        let logit1 = dot1 / t_val;
        let max_logit = logit0.max(logit1);
        let exp0 = (logit0 - max_logit).exp();
        let exp1 = (logit1 - max_logit).exp();
        let sum_exp = exp0 + exp1;
        let soft0 = exp0 / sum_exp;
        let soft1 = exp1 / sum_exp;

        let y_val = soft0 * dot0 + soft1 * dot1;
        Tensor1D::from_scalar(y_val)
    }
}