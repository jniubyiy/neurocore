use crate::tensor::Tensor3D;
use crate::model_plan::param_store::ParamSlice;
use crate::neuron::Memory as MemoryNeuron;
use crate::neuron::base::Neuron;
use crate::linalg;
use super::{Layer3D, LayerContext3D};

pub struct Memory3D {
    pub dim3: usize,
    pub out_dim3: usize,
}

impl Memory3D {
    pub fn new(dim3: usize, out_dim3: usize) -> Self {
        Self { dim3, out_dim3 }
    }
}

impl Layer3D for Memory3D {
    fn input_dims(&self) -> Vec<usize> { vec![self.dim3] }
    fn output_dims(&self) -> Vec<usize> { vec![self.out_dim3] }

    fn forward_into(
        &self,
        inputs: &[Tensor3D],
        params: &[f32],
        slice: &ParamSlice,
        out_bufs: &mut [Vec<Vec<Vec<f32>>>],
    ) -> Vec<LayerContext3D> {
        assert_eq!(inputs.len(), 1);
        assert_eq!(out_bufs.len(), 1);
        let input = &inputs[0];
        let out_buf = &mut out_bufs[0];
        let dim1 = input.dim1;
        let dim2 = input.dim2;
        let dim3 = input.dim3;
        assert_eq!(dim3, self.dim3);
        assert_eq!(out_buf.len(), dim1);
        assert_eq!(out_buf[0].len(), dim2);
        assert_eq!(out_buf[0][0].len(), self.out_dim3);

        let x = linalg::tensor3d_to_faer(input);

        for out_i in 0..self.out_dim3 {
            let offset = slice.start + out_i * (2 * self.dim3 + 1);
            let m0 = params[offset .. offset + self.dim3].to_vec();
            let m1 = params[offset + self.dim3 .. offset + 2 * self.dim3].to_vec();
            let t_val = params[offset + 2 * self.dim3];

            let neuron = MemoryNeuron::new(m0, m1, t_val);
            let result_col = neuron.forward_mat(&x);

            let mut idx = 0;
            for i in 0..dim1 {
                for j in 0..dim2 {
                    out_buf[i][j][out_i] = result_col[(idx, 0)];
                    idx += 1;
                }
            }
        }

        vec![LayerContext3D::Memory3D { input: input.clone() }]
    }

    fn backward(
        &self,
        ctxs: &[LayerContext3D],
        deltas: &[Tensor3D],
        params: &[f32],
        slice: &ParamSlice,
    ) -> (Vec<Tensor3D>, Vec<f32>) {
        assert_eq!(ctxs.len(), 1);
        assert_eq!(deltas.len(), 1);
        let ctx = &ctxs[0];
        let delta = &deltas[0];

        let input = match ctx {
            LayerContext3D::Memory3D { input } => input,
            _ => panic!("Invalid context for Memory3D"),
        };
        let dim1 = input.dim1;
        let dim2 = input.dim2;
        let dim3 = input.dim3;
        assert_eq!(delta.dim1, dim1);
        assert_eq!(delta.dim2, dim2);
        assert_eq!(delta.dim3, self.out_dim3);

        let mut delta_prev = vec![vec![vec![0.0; dim3]; dim2]; dim1];
        let mut grad = vec![0.0; self.param_len()];

        for i in 0..dim1 {
            for j in 0..dim2 {
                for out_i in 0..self.out_dim3 {
                    let offset = slice.start + out_i * (2 * dim3 + 1);
                    let m0 = &params[offset .. offset + dim3];
                    let m1 = &params[offset + dim3 .. offset + 2 * dim3];
                    let t_val = params[offset + 2 * dim3];

                    let mut dot0 = 0.0;
                    let mut dot1 = 0.0;
                    for k in 0..dim3 {
                        dot0 += input.data[i][j][k] * m0[k];
                        dot1 += input.data[i][j][k] * m1[k];
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

                    let delta_val = delta.data[i][j][out_i];

                    let base = out_i * (2 * dim3 + 1);
                    for k in 0..dim3 {
                        grad[base + k] += delta_val * dy_dot0 * input.data[i][j][k];
                        grad[base + dim3 + k] += delta_val * dy_dot1 * input.data[i][j][k];
                    }
                    let avg_dot = soft0 * dot0 + soft1 * dot1;
                    let ds0_dt = soft0 * (dot0 - avg_dot) / (t_val * t_val);
                    let ds1_dt = soft1 * (dot1 - avg_dot) / (t_val * t_val);
                    let dy_dt = dot0 * ds0_dt + dot1 * ds1_dt;
                    grad[base + 2 * dim3] += delta_val * dy_dt;

                    for k in 0..dim3 {
                        delta_prev[i][j][k] += delta_val * (dy_dot0 * m0[k] + dy_dot1 * m1[k]);
                    }
                }
            }
        }

        (vec![Tensor3D::new(delta_prev)], grad)
    }

    fn param_len(&self) -> usize {
        self.out_dim3 * (2 * self.dim3 + 1)
    }
}