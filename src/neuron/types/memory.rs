use crate::tensor::Tensor1D;
use crate::jacobian::Jacobian;
use crate::neuron::base::Neuron;

pub struct Memory {
    memory0: Vec<f32>,
    memory1: Vec<f32>,
    temperature: f32,
}

impl Memory {
    pub fn new(input_dim: usize, _start_index: usize) -> Self {
        Memory {
            memory0: vec![0.0; input_dim],
            memory1: vec![0.0; input_dim],
            temperature: 1.0,
        }
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
        self.memory0.copy_from_slice(&values[..n]);
        self.memory1.copy_from_slice(&values[n..2 * n]);
        self.temperature = values[2 * n];
    }
}

impl Neuron for Memory {
    fn forward(&self, input: &Tensor1D, j_input: &Jacobian) -> (Tensor1D, Jacobian) {
        let n = input.len();
        assert_eq!(n, self.memory0.len());
        let p = j_input.num_params;          // <-- исправлено

        let t_val = self.temperature;
        let m0 = &self.memory0;
        let m1 = &self.memory1;

        let mut dot0 = 0.0;
        let mut dot1 = 0.0;
        for i in 0..n {
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

        let ds0_dot0 = soft0 * (1.0 - soft0) / t_val;
        let ds0_dot1 = -soft0 * soft1 / t_val;
        let ds1_dot0 = -soft1 * soft0 / t_val;
        let ds1_dot1 = soft1 * (1.0 - soft1) / t_val;

        let dy_dot0 = soft0 + dot0 * ds0_dot0 + dot1 * ds1_dot0;
        let dy_dot1 = soft1 + dot0 * ds0_dot1 + dot1 * ds1_dot1;

        let mut grad = vec![0.0_f32; p];
        for k in 0..n {
            let dy_dxk = dy_dot0 * m0[k] + dy_dot1 * m1[k];
            for j in 0..p {
                grad[j] += dy_dxk * j_input.data[k][j];
            }
        }

        (Tensor1D::from_scalar(y_val), Jacobian {
            out_features: 1,
            num_params: p,
            data: vec![grad],
        })
    }
}