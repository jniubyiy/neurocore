use crate::tensor::Tensor2D;
use crate::model_plan::param_store::ParamSlice;
use crate::neuron::Memory as MemoryNeuron;
use crate::neuron::base::Neuron;
use crate::linalg;
use super::{Layer2D, LayerContext};

pub struct Memory2D {
    pub input_dim: usize,
    pub output_dim: usize,
}

impl Memory2D {
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        Self { input_dim, output_dim }
    }
}

impl Layer2D for Memory2D {
    fn forward_into(&self, input: &Tensor2D, params: &[f32], slice: &ParamSlice, out_buf: &mut Vec<Vec<f32>>) -> LayerContext {
        let rows = input.rows;
        assert_eq!(input.cols, self.input_dim);
        assert_eq!(out_buf.len(), rows);
        assert_eq!(out_buf[0].len(), self.output_dim);

        let x = linalg::tensor2d_to_faer(input);

        for out_i in 0..self.output_dim {
            let offset = slice.start + out_i * (2 * self.input_dim + 1);
            let m0 = params[offset .. offset + self.input_dim].to_vec();
            let m1 = params[offset + self.input_dim .. offset + 2 * self.input_dim].to_vec();
            let t_val = params[offset + 2 * self.input_dim];

            let neuron = MemoryNeuron::new(m0, m1, t_val);
            let result_col = neuron.forward_mat(&x);

            for r in 0..rows {
                out_buf[r][out_i] = result_col[(r, 0)];
            }
        }

        LayerContext::Linear2D { input: input.clone() }
    }

    fn backward(&self, ctx: &LayerContext, delta: &Tensor2D, params: &[f32], slice: &ParamSlice) -> (Tensor2D, Vec<f32>) {
        let input = match ctx {
            LayerContext::Linear2D { input } => input,
            _ => panic!("Invalid context for Memory2D"),
        };
        let rows = input.rows;
        let mut delta_prev = vec![vec![0.0; self.input_dim]; rows];
        let mut grad = vec![0.0; self.param_len()];

        for r in 0..rows {
            for out_i in 0..self.output_dim {
                let offset = slice.start + out_i * (2 * self.input_dim + 1);
                let m0 = &params[offset .. offset + self.input_dim];
                let m1 = &params[offset + self.input_dim .. offset + 2 * self.input_dim];
                let t_val = params[offset + 2 * self.input_dim];

                let mut dot0 = 0.0;
                let mut dot1 = 0.0;
                for i in 0..self.input_dim {
                    dot0 += input.data[r][i] * m0[i];
                    dot1 += input.data[r][i] * m1[i];
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

                let delta_val = delta.data[r][out_i];

                let base_idx = out_i * (2 * self.input_dim + 1);
                for i in 0..self.input_dim {
                    grad[base_idx + i] += delta_val * (dy_dot0 * input.data[r][i]);
                    grad[base_idx + self.input_dim + i] += delta_val * (dy_dot1 * input.data[r][i]);
                }

                let avg_dot = soft0 * dot0 + soft1 * dot1;
                let ds0_dt = soft0 * (dot0 - avg_dot) / (t_val * t_val);
                let ds1_dt = soft1 * (dot1 - avg_dot) / (t_val * t_val);
                let dy_dt = dot0 * ds0_dt + dot1 * ds1_dt;
                grad[base_idx + 2 * self.input_dim] += delta_val * dy_dt;

                for i in 0..self.input_dim {
                    delta_prev[r][i] += delta_val * (dy_dot0 * m0[i] + dy_dot1 * m1[i]);
                }
            }
        }

        (Tensor2D::new(delta_prev), grad)
    }

    fn param_len(&self) -> usize {
        self.output_dim * (2 * self.input_dim + 1)
    }

    fn in_features(&self) -> usize { self.input_dim }
    fn out_features(&self) -> usize { self.output_dim }
}