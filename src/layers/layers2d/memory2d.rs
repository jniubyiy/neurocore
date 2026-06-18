use crate::tensor::Tensor2D;
use crate::model_plan::param_store::ParamSlice;
use crate::neuron::Memory as MemoryNeuron;
use crate::neuron::base::Neuron;
use crate::linalg;
use super::{Layer2D, LayerContext};

pub struct Memory2D {
    pub dim2: usize,
    pub out_dim2: usize,
}

impl Memory2D {
    pub fn new(dim2: usize, out_dim2: usize) -> Self {
        Self { dim2, out_dim2 }
    }
}

impl Layer2D for Memory2D {
    fn input_dims(&self) -> Vec<usize> { vec![self.dim2] }
    fn output_dims(&self) -> Vec<usize> { vec![self.out_dim2] }

    fn forward_into(
        &self,
        inputs: &[Tensor2D],
        params: &[f32],
        slice: &ParamSlice,
        out_bufs: &mut [Vec<Vec<f32>>],
    ) -> Vec<LayerContext> {
        assert_eq!(inputs.len(), 1);
        assert_eq!(out_bufs.len(), 1);
        let input = &inputs[0];
        let out_buf = &mut out_bufs[0];
        let dim1 = input.dim1;
        let dim2 = input.dim2;
        assert_eq!(dim2, self.dim2);
        assert_eq!(out_buf.len(), dim1);
        assert_eq!(out_buf[0].len(), self.out_dim2);

        let x = linalg::tensor2d_to_faer(input);

        for out_i in 0..self.out_dim2 {
            let offset = slice.start + out_i * (2 * self.dim2 + 1);
            let m0 = params[offset .. offset + self.dim2].to_vec();
            let m1 = params[offset + self.dim2 .. offset + 2 * self.dim2].to_vec();
            let t_val = params[offset + 2 * self.dim2];

            let neuron = MemoryNeuron::new(m0, m1, t_val);
            let result_col = neuron.forward_mat(&x);

            for r in 0..dim1 {
                out_buf[r][out_i] = result_col[(r, 0)];
            }
        }

        vec![LayerContext::Memory2D { input: input.clone() }]
    }

    fn backward(
        &self,
        ctxs: &[LayerContext],
        deltas: &[Tensor2D],
        params: &[f32],
        slice: &ParamSlice,
    ) -> (Vec<Tensor2D>, Vec<f32>) {
        assert_eq!(ctxs.len(), 1);
        assert_eq!(deltas.len(), 1);
        let ctx = &ctxs[0];
        let delta = &deltas[0];

        let input = match ctx {
            LayerContext::Memory2D { input } => input,
            _ => panic!("Invalid context for Memory2D"),
        };
        let dim1 = input.dim1;
        assert_eq!(delta.dim1, dim1);
        assert_eq!(delta.dim2, self.out_dim2);

        let mut delta_prev = vec![vec![0.0; self.dim2]; dim1];
        let mut grad = vec![0.0; self.param_len()];

        for r in 0..dim1 {
            for out_i in 0..self.out_dim2 {
                let offset = slice.start + out_i * (2 * self.dim2 + 1);
                let m0 = &params[offset .. offset + self.dim2];
                let m1 = &params[offset + self.dim2 .. offset + 2 * self.dim2];
                let t_val = params[offset + 2 * self.dim2];

                let mut dot0 = 0.0;
                let mut dot1 = 0.0;
                for i in 0..self.dim2 {
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

                let base_idx = out_i * (2 * self.dim2 + 1);
                for i in 0..self.dim2 {
                    grad[base_idx + i] += delta_val * (dy_dot0 * input.data[r][i]);
                    grad[base_idx + self.dim2 + i] += delta_val * (dy_dot1 * input.data[r][i]);
                }

                let avg_dot = soft0 * dot0 + soft1 * dot1;
                let ds0_dt = soft0 * (dot0 - avg_dot) / (t_val * t_val);
                let ds1_dt = soft1 * (dot1 - avg_dot) / (t_val * t_val);
                let dy_dt = dot0 * ds0_dt + dot1 * ds1_dt;
                grad[base_idx + 2 * self.dim2] += delta_val * dy_dt;

                for i in 0..self.dim2 {
                    delta_prev[r][i] += delta_val * (dy_dot0 * m0[i] + dy_dot1 * m1[i]);
                }
            }
        }

        (vec![Tensor2D::new(delta_prev)], grad)
    }

    fn param_len(&self) -> usize {
        self.out_dim2 * (2 * self.dim2 + 1)
    }
}