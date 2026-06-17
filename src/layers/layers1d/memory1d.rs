use crate::tensor::Tensor1D;
use crate::model_plan::param_store::{ParamSlice, ParamStore};
use crate::neuron::Memory as MemoryNeuron;
use crate::neuron::base::Neuron;
use crate::linalg;
use super::{Layer, LayerInfo};

pub struct MemoryLayer {
    input_dim: usize,
    output_dim: usize,
    last_input: Option<Tensor1D>,
    grad_m0: Vec<f32>,
    grad_m1: Vec<f32>,
    grad_t: Vec<f32>,
}

impl MemoryLayer {
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        assert!(input_dim > 0 && output_dim > 0);
        Self {
            input_dim,
            output_dim,
            last_input: None,
            grad_m0: vec![0.0; output_dim * input_dim],
            grad_m1: vec![0.0; output_dim * input_dim],
            grad_t: vec![0.0; output_dim],
        }
    }
}

impl Layer for MemoryLayer {
    fn forward_into(&mut self, input: &Tensor1D, params: &[f32], slice: &ParamSlice, out_buf: &mut Vec<f32>) {
        assert_eq!(input.len(), self.input_dim);
        assert_eq!(out_buf.len(), self.output_dim);
        self.last_input = Some(input.clone());

        let input_mat = linalg::tensor1d_to_faer(input);

        for out_i in 0..self.output_dim {
            let offset = slice.start + out_i * (2 * self.input_dim + 1);
            let m0 = params[offset .. offset + self.input_dim].to_vec();
            let m1 = params[offset + self.input_dim .. offset + 2 * self.input_dim].to_vec();
            let t_val = params[offset + 2 * self.input_dim];

            let neuron = MemoryNeuron::new(m0, m1, t_val);
            let result_mat = neuron.forward_mat(&input_mat);
            out_buf[out_i] = result_mat[(0, 0)];
        }
    }

    fn backward(&mut self, delta: &Tensor1D, params: &[f32], slice: &ParamSlice) -> Tensor1D {
        let input = self.last_input.take().expect("Memory backward without forward");
        assert_eq!(delta.len(), self.output_dim);

        let mut delta_prev = vec![0.0; self.input_dim];

        for out_i in 0..self.output_dim {
            let offset = slice.start + out_i * (2 * self.input_dim + 1);
            let m0 = &params[offset .. offset + self.input_dim];
            let m1 = &params[offset + self.input_dim .. offset + 2 * self.input_dim];
            let t_val = params[offset + 2 * self.input_dim];

            let mut dot0 = 0.0;
            let mut dot1 = 0.0;
            for i in 0..self.input_dim {
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

            let ds0_dot0 = soft0 * (1.0 - soft0) / t_val;
            let ds0_dot1 = -soft0 * soft1 / t_val;
            let ds1_dot0 = -soft1 * soft0 / t_val;
            let ds1_dot1 = soft1 * (1.0 - soft1) / t_val;

            let dy_dot0 = soft0 + dot0 * ds0_dot0 + dot1 * ds1_dot0;
            let dy_dot1 = soft1 + dot0 * ds0_dot1 + dot1 * ds1_dot1;

            let delta_val = delta.data[out_i];

            for i in 0..self.input_dim {
                self.grad_m0[out_i * self.input_dim + i] += delta_val * (dy_dot0 * input.data[i]);
                self.grad_m1[out_i * self.input_dim + i] += delta_val * (dy_dot1 * input.data[i]);
            }

            let avg_dot = soft0 * dot0 + soft1 * dot1;
            let ds0_dt = soft0 * (dot0 - avg_dot) / (t_val * t_val);
            let ds1_dt = soft1 * (dot1 - avg_dot) / (t_val * t_val);
            let dy_dt = dot0 * ds0_dt + dot1 * ds1_dt;
            self.grad_t[out_i] += delta_val * dy_dt;

            for i in 0..self.input_dim {
                delta_prev[i] += delta_val * (dy_dot0 * m0[i] + dy_dot1 * m1[i]);
            }
        }

        Tensor1D::new(delta_prev)
    }

    fn apply_gradients(&mut self, store: &mut ParamStore, lr: f32, slice: &ParamSlice) {
        for out_i in 0..self.output_dim {
            let base = slice.start + out_i * (2 * self.input_dim + 1);
            for i in 0..self.input_dim {
                let idx = base + i;
                store.add_to_param(idx, -lr * self.grad_m0[out_i * self.input_dim + i]);
            }
            for i in 0..self.input_dim {
                let idx = base + self.input_dim + i;
                store.add_to_param(idx, -lr * self.grad_m1[out_i * self.input_dim + i]);
            }
            let idx = base + 2 * self.input_dim;
            store.add_to_param(idx, -lr * self.grad_t[out_i]);
        }
        self.grad_m0.fill(0.0);
        self.grad_m1.fill(0.0);
        self.grad_t.fill(0.0);
    }

    fn param_len(&self) -> usize {
        self.output_dim * (2 * self.input_dim + 1)
    }

    fn layer_info(&self) -> LayerInfo {
        LayerInfo {
            layer_type: "Memory".to_string(),
            input_dim: self.input_dim,
            output_dim: self.output_dim,
            param_count: self.param_len(),
            param_start_index: None,
        }
    }
}