use crate::tensor::Tensor5D;
use crate::model_plan::param_store::ParamSlice;
use crate::neuron::Memory as MemoryNeuron;
use crate::neuron::base::Neuron;
use crate::linalg;
use super::{Layer5D, LayerContext5D};

pub struct Memory5D {
    pub dim5: usize,
    pub out_dim5: usize,
}

impl Memory5D {
    pub fn new(dim5: usize, out_dim5: usize) -> Self {
        Self { dim5, out_dim5 }
    }
}

impl Layer5D for Memory5D {
    fn input_dims(&self) -> Vec<usize> { vec![self.dim5] }
    fn output_dims(&self) -> Vec<usize> { vec![self.out_dim5] }

    fn forward_into(
        &self,
        inputs: &[Tensor5D],
        params: &[f32],
        slice: &ParamSlice,
        out_bufs: &mut [Vec<Vec<Vec<Vec<Vec<f32>>>>>],
    ) -> Vec<LayerContext5D> {
        assert_eq!(inputs.len(), 1);
        assert_eq!(out_bufs.len(), 1);
        let input = &inputs[0];
        let out_buf = &mut out_bufs[0];
        let dim1 = input.dim1;
        let dim2 = input.dim2;
        let dim3 = input.dim3;
        let dim4 = input.dim4;
        let dim5 = input.dim5;
        assert_eq!(dim5, self.dim5);
        assert_eq!(out_buf.len(), dim1);
        assert_eq!(out_buf[0].len(), dim2);
        assert_eq!(out_buf[0][0].len(), dim3);
        assert_eq!(out_buf[0][0][0].len(), dim4);
        assert_eq!(out_buf[0][0][0][0].len(), self.out_dim5);

        let x = linalg::tensor5d_to_faer(input);

        for out_i in 0..self.out_dim5 {
            let offset = slice.start + out_i * (2 * self.dim5 + 1);
            let m0 = params[offset .. offset + self.dim5].to_vec();
            let m1 = params[offset + self.dim5 .. offset + 2 * self.dim5].to_vec();
            let t_val = params[offset + 2 * self.dim5];

            let neuron = MemoryNeuron::new(m0, m1, t_val);
            let result_col = neuron.forward_mat(&x);

            let mut idx = 0;
            for i in 0..dim1 {
                for j in 0..dim2 {
                    for k in 0..dim3 {
                        for l in 0..dim4 {
                            out_buf[i][j][k][l][out_i] = result_col[(idx, 0)];
                            idx += 1;
                        }
                    }
                }
            }
        }

        vec![LayerContext5D::Memory5D { input: input.clone() }]
    }

    fn backward(
        &self,
        ctxs: &[LayerContext5D],
        deltas: &[Tensor5D],
        params: &[f32],
        slice: &ParamSlice,
    ) -> (Vec<Tensor5D>, Vec<f32>) {
        assert_eq!(ctxs.len(), 1);
        assert_eq!(deltas.len(), 1);
        let ctx = &ctxs[0];
        let delta = &deltas[0];

        let input = match ctx {
            LayerContext5D::Memory5D { input } => input,
            _ => panic!("Invalid context for Memory5D"),
        };
        let dim1 = input.dim1;
        let dim2 = input.dim2;
        let dim3 = input.dim3;
        let dim4 = input.dim4;
        let dim5 = input.dim5;
        assert_eq!(delta.dim1, dim1);
        assert_eq!(delta.dim2, dim2);
        assert_eq!(delta.dim3, dim3);
        assert_eq!(delta.dim4, dim4);
        assert_eq!(delta.dim5, self.out_dim5);

        let mut delta_prev = vec![vec![vec![vec![vec![0.0; dim5]; dim4]; dim3]; dim2]; dim1];
        let mut grad = vec![0.0; self.param_len()];

        for i in 0..dim1 {
            for j in 0..dim2 {
                for k in 0..dim3 {
                    for l in 0..dim4 {
                        for out_i in 0..self.out_dim5 {
                            let offset = slice.start + out_i * (2 * dim5 + 1);
                            let m0 = &params[offset .. offset + dim5];
                            let m1 = &params[offset + dim5 .. offset + 2 * dim5];
                            let t_val = params[offset + 2 * dim5];

                            let mut dot0 = 0.0;
                            let mut dot1 = 0.0;
                            for n in 0..dim5 {
                                dot0 += input.data[i][j][k][l][n] * m0[n];
                                dot1 += input.data[i][j][k][l][n] * m1[n];
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

                            let delta_val = delta.data[i][j][k][l][out_i];

                            let base = out_i * (2 * dim5 + 1);
                            for n in 0..dim5 {
                                grad[base + n] += delta_val * dy_dot0 * input.data[i][j][k][l][n];
                                grad[base + dim5 + n] += delta_val * dy_dot1 * input.data[i][j][k][l][n];
                            }
                            let avg_dot = soft0 * dot0 + soft1 * dot1;
                            let ds0_dt = soft0 * (dot0 - avg_dot) / (t_val * t_val);
                            let ds1_dt = soft1 * (dot1 - avg_dot) / (t_val * t_val);
                            let dy_dt = dot0 * ds0_dt + dot1 * ds1_dt;
                            grad[base + 2 * dim5] += delta_val * dy_dt;

                            for n in 0..dim5 {
                                delta_prev[i][j][k][l][n] += delta_val * (dy_dot0 * m0[n] + dy_dot1 * m1[n]);
                            }
                        }
                    }
                }
            }
        }

        (vec![Tensor5D::new(delta_prev)], grad)
    }

    fn param_len(&self) -> usize {
        self.out_dim5 * (2 * self.dim5 + 1)
    }
}