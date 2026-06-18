use crate::tensor::Tensor1D;
use crate::model_plan::param_store::ParamSlice;
use crate::neuron::Memory as MemoryNeuron;
use crate::neuron::base::Neuron;
use crate::linalg;
use super::{Layer, LayerContext1D, LayerInfo};

pub struct MemoryLayer {
    pub dim1: usize,
    pub out_dim1: usize,
}

impl MemoryLayer {
    pub fn new(dim1: usize, out_dim1: usize) -> Self {
        assert!(dim1 > 0 && out_dim1 > 0);
        Self { dim1, out_dim1 }
    }
}

impl Layer for MemoryLayer {
    fn input_dim1s(&self) -> Vec<usize> { vec![self.dim1] }
    fn output_dim1s(&self) -> Vec<usize> { vec![self.out_dim1] }

    fn forward_into(
        &self,
        inputs: &[Tensor1D],
        params: &[f32],
        slice: &ParamSlice,
        out_bufs: &mut [Vec<f32>],
    ) -> Vec<LayerContext1D> {
        assert_eq!(inputs.len(), 1);
        assert_eq!(out_bufs.len(), 1);
        let input = &inputs[0];
        assert_eq!(input.dim1(), self.dim1);
        let out_buf = &mut out_bufs[0];
        assert_eq!(out_buf.len(), self.out_dim1);

        let input_mat = linalg::tensor1d_to_faer(input);
        for out_i in 0..self.out_dim1 {
            let offset = slice.start + out_i * (2 * self.dim1 + 1);
            let m0 = params[offset .. offset + self.dim1].to_vec();
            let m1 = params[offset + self.dim1 .. offset + 2 * self.dim1].to_vec();
            let t_val = params[offset + 2 * self.dim1];

            let neuron = MemoryNeuron::new(m0, m1, t_val);
            let result_mat = neuron.forward_mat(&input_mat);
            out_buf[out_i] = result_mat[(0, 0)];
        }

        vec![LayerContext1D::Memory { input: input.clone() }]
    }

    fn backward(
        &self,
        ctxs: &[LayerContext1D],
        deltas: &[Tensor1D],
        params: &[f32],
        slice: &ParamSlice,
    ) -> (Vec<Tensor1D>, Vec<f32>) {
        assert_eq!(ctxs.len(), 1);
        assert_eq!(deltas.len(), 1);
        let ctx = &ctxs[0];
        let delta = &deltas[0];

        let input = match ctx {
            LayerContext1D::Memory { input } => input,
            _ => panic!(),
        };
        assert_eq!(delta.dim1(), self.out_dim1);

        let mut grad = vec![0.0; self.param_len()];
        let mut delta_prev = vec![0.0; self.dim1];

        for out_i in 0..self.out_dim1 {
            let offset = slice.start + out_i * (2 * self.dim1 + 1);
            let m0 = &params[offset .. offset + self.dim1];
            let m1 = &params[offset + self.dim1 .. offset + 2 * self.dim1];
            let t_val = params[offset + 2 * self.dim1];

            let mut dot0 = 0.0;
            let mut dot1 = 0.0;
            for i in 0..self.dim1 {
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

            let base = out_i * (2 * self.dim1 + 1);
            for i in 0..self.dim1 {
                grad[base + i] += delta_val * dy_dot0 * input.data[i];
                grad[base + self.dim1 + i] += delta_val * dy_dot1 * input.data[i];
            }
            let avg_dot = soft0 * dot0 + soft1 * dot1;
            let ds0_dt = soft0 * (dot0 - avg_dot) / (t_val * t_val);
            let ds1_dt = soft1 * (dot1 - avg_dot) / (t_val * t_val);
            let dy_dt = dot0 * ds0_dt + dot1 * ds1_dt;
            grad[base + 2 * self.dim1] += delta_val * dy_dt;

            for i in 0..self.dim1 {
                delta_prev[i] += delta_val * (dy_dot0 * m0[i] + dy_dot1 * m1[i]);
            }
        }

        (vec![Tensor1D::new(delta_prev)], grad)
    }

    fn param_len(&self) -> usize {
        self.out_dim1 * (2 * self.dim1 + 1)
    }

    fn layer_info(&self) -> LayerInfo {
        LayerInfo {
            layer_type: "Memory".to_string(),
            input_dim1s: self.input_dim1s(),
            output_dim1s: self.output_dim1s(),
            param_count: self.param_len(),
            param_start_index: None,
        }
    }
}