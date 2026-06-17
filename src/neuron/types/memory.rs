use faer::Mat;
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

    pub fn param_count(&self) -> usize { 2 * self.memory0.len() + 1 }
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

    fn mem_col(&self, data: &[f32]) -> Mat<f32> {
        Mat::from_fn(data.len(), 1, |i, _| data[i])
    }
}

impl Neuron for Memory {
    fn apply(&self, _x: f32) -> f32 {
        panic!("Memory does not support element‑wise apply; use forward()");
    }

    fn forward(&self, input: &crate::tensor::Tensor1D) -> crate::tensor::Tensor1D {
        let mut dot0 = 0.0;
        let mut dot1 = 0.0;
        for i in 0..input.len() {
            dot0 += input.data[i] * self.memory0[i];
            dot1 += input.data[i] * self.memory1[i];
        }
        let logit0 = dot0 / self.temperature;
        let logit1 = dot1 / self.temperature;
        let max_logit = logit0.max(logit1);
        let exp0 = (logit0 - max_logit).exp();
        let exp1 = (logit1 - max_logit).exp();
        let sum_exp = exp0 + exp1;
        let soft0 = exp0 / sum_exp;
        let soft1 = exp1 / sum_exp;
        crate::tensor::Tensor1D::from_scalar(soft0 * dot0 + soft1 * dot1)
    }

    fn forward_mat(&self, input: &Mat<f32>) -> Mat<f32> {
        let batch = input.nrows();
        let m0_col = self.mem_col(&self.memory0);
        let m1_col = self.mem_col(&self.memory1);
        let dot0 = input * &m0_col;
        let dot1 = input * &m1_col;

        let mut result = Mat::zeros(batch, 1);
        for i in 0..batch {
            let d0 = dot0[(i, 0)] / self.temperature;
            let d1 = dot1[(i, 0)] / self.temperature;
            let max_logit = d0.max(d1);
            let exp0 = (d0 - max_logit).exp();
            let exp1 = (d1 - max_logit).exp();
            let sum = exp0 + exp1;
            result[(i, 0)] = (exp0 / sum) * dot0[(i, 0)] + (exp1 / sum) * dot1[(i, 0)];
        }
        result
    }
}