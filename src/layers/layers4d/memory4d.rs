use crate::tensor::Tensor4D;
use crate::model_plan::param_store::ParamSlice;
use crate::neuron::Memory as MemoryNeuron;
use crate::neuron::base::Neuron;
use crate::linalg;
use super::{Layer4D, LayerContext4D};

pub struct Memory4D {
    pub dim4: usize,
    pub out_dim4: usize,
}

impl Memory4D {
    pub fn new(dim4: usize, out_dim4: usize) -> Self {
        Self { dim4, out_dim4 }
    }
}

impl Layer4D for Memory4D {
    fn input_dims(&self) -> Vec<usize> { vec![self.dim4] }
    fn output_dims(&self) -> Vec<usize> { vec![self.out_dim4] }

    fn forward_into(
        &self,
        inputs: &[Tensor4D],
        params: &[f32],
        slice: &ParamSlice,
        out_bufs: &mut [Vec<Vec<Vec<Vec<f32>>>>],
    ) -> Vec<LayerContext4D> {
        assert_eq!(inputs.len(), 1);
        assert_eq!(out_bufs.len(), 1);
        let input = &inputs[0];
        let out_buf = &mut out_bufs[0];
        let dim1 = input.dim1;
        let dim2 = input.dim2;
        let dim3 = input.dim3;
        let dim4 = input.dim4;
        assert_eq!(dim4, self.dim4);
        assert_eq!(out_buf.len(), dim1);
        assert_eq!(out_buf[0].len(), dim2);
        assert_eq!(out_buf[0][0].len(), dim3);
        assert_eq!(out_buf[0][0][0].len(), self.out_dim4);

        let x = linalg::tensor4d_to_faer(input);

        for out_i in 0..self.out_dim4 {
            let offset = slice.start + out_i * (2 * self.dim4 + 1);
            let m0 = params[offset .. offset + self.dim4].to_vec();
            let m1 = params[offset + self.dim4 .. offset + 2 * self.dim4].to_vec();
            let t_val = params[offset + 2 * self.dim4];

            let neuron = MemoryNeuron::new(m0, m1, t_val);
            let result_col = neuron.forward_mat(&x);

            let mut idx = 0;
            for i in 0..dim1 {
                for j in 0..dim2 {
                    for k in 0..dim3 {
                        out_buf[i][j][k][out_i] = result_col[(idx, 0)];
                        idx += 1;
                    }
                }
            }
        }

        vec![LayerContext4D::Memory4D { input: input.clone() }]
    }

    fn backward(
        &self,
        ctxs: &[LayerContext4D],
        deltas: &[Tensor4D],
        params: &[f32],
        slice: &ParamSlice,
    ) -> (Vec<Tensor4D>, Vec<f32>) {
        assert_eq!(ctxs.len(), 1);
        assert_eq!(deltas.len(), 1);
        let ctx = &ctxs[0];
        let delta = &deltas[0];

        let input = match ctx {
            LayerContext4D::Memory4D { input } => input,
            _ => panic!("Invalid context for Memory4D"),
        };
        let dim1 = input.dim1;
        let dim2 = input.dim2;
        let dim3 = input.dim3;
        let dim4 = input.dim4;
        assert_eq!(delta.dim1, dim1);
        assert_eq!(delta.dim2, dim2);
        assert_eq!(delta.dim3, dim3);
        assert_eq!(delta.dim4, self.out_dim4);

        let mut delta_prev = vec![vec![vec![vec![0.0; dim4]; dim3]; dim2]; dim1];
        let mut grad = vec![0.0; self.param_len()];

        for i in 0..dim1 {
            for j in 0..dim2 {
                for k in 0..dim3 {
                    for out_i in 0..self.out_dim4 {
                        let offset = slice.start + out_i * (2 * dim4 + 1);
                        let m0 = &params[offset .. offset + dim4];
                        let m1 = &params[offset + dim4 .. offset + 2 * dim4];
                        let t_val = params[offset + 2 * dim4];

                        let mut dot0 = 0.0;
                        let mut dot1 = 0.0;
                        for n in 0..dim4 {
                            dot0 += input.data[i][j][k][n] * m0[n];
                            dot1 += input.data[i][j][k][n] * m1[n];
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

                        let delta_val = delta.data[i][j][k][out_i];

                        let base = out_i * (2 * dim4 + 1);
                        for n in 0..dim4 {
                            grad[base + n] += delta_val * dy_dot0 * input.data[i][j][k][n];
                            grad[base + dim4 + n] += delta_val * dy_dot1 * input.data[i][j][k][n];
                        }
                        let avg_dot = soft0 * dot0 + soft1 * dot1;
                        let ds0_dt = soft0 * (dot0 - avg_dot) / (t_val * t_val);
                        let ds1_dt = soft1 * (dot1 - avg_dot) / (t_val * t_val);
                        let dy_dt = dot0 * ds0_dt + dot1 * ds1_dt;
                        grad[base + 2 * dim4] += delta_val * dy_dt;

                        for n in 0..dim4 {
                            delta_prev[i][j][k][n] += delta_val * (dy_dot0 * m0[n] + dy_dot1 * m1[n]);
                        }
                    }
                }
            }
        }

        (vec![Tensor4D::new(delta_prev)], grad)
    }

    fn param_len(&self) -> usize {
        self.out_dim4 * (2 * self.dim4 + 1)
    }
}