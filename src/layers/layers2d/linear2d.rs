use crate::tensor::Tensor2D;
use crate::model_plan::param_store::ParamSlice;
use crate::model_plan::blueprint::assert_power_of_two;
use crate::neuron::Linear;
use super::{Layer2D, LayerContext};

pub struct Linear2D {
    pub input_dim: usize,
    pub output_dim: usize,
}

impl Linear2D {
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        assert_power_of_two(input_dim);
        assert_power_of_two(output_dim);
        Self { input_dim, output_dim }
    }

    fn weight_index(&self, out_idx: usize, in_idx: usize, slice: &ParamSlice) -> usize {
        slice.start + out_idx * self.input_dim + in_idx
    }

    fn bias_index(&self, out_idx: usize, slice: &ParamSlice) -> usize {
        slice.start + self.input_dim * self.output_dim + out_idx
    }
}

impl Layer2D for Linear2D {
    fn forward_into(&self, input: &Tensor2D, params: &[f32], slice: &ParamSlice, out_buf: &mut Vec<Vec<f32>>) -> LayerContext {
        assert_eq!(input.cols, self.input_dim);
        assert_eq!(out_buf.len(), input.rows);
        assert_eq!(out_buf[0].len(), self.output_dim);

        for r in 0..input.rows {
            for o in 0..self.output_dim {
                let w_start = self.weight_index(o, 0, slice);
                let w_end = w_start + self.input_dim;
                let weights = &params[w_start..w_end];
                let bias = params[self.bias_index(o, slice)];
                out_buf[r][o] = Linear::forward_slice(weights, bias, &input.data[r]);
            }
        }

        LayerContext::Linear2D { input: input.clone() }
    }

    fn backward(&self, ctx: &LayerContext, delta: &Tensor2D, params: &[f32], slice: &ParamSlice) -> (Tensor2D, Vec<f32>) {
        let input = match ctx {
            LayerContext::Linear2D { input } => input,
            _ => panic!("Invalid context for Linear2D"),
        };
        let rows = input.rows;
        let mut delta_prev = vec![vec![0.0; self.input_dim]; rows];
        let mut grad = vec![0.0; self.param_len()];

        for r in 0..rows {
            for o in 0..self.output_dim {
                let d = delta.data[r][o];
                for i in 0..self.input_dim {
                    let idx = o * self.input_dim + i;
                    grad[idx] += d * input.data[r][i];
                }
                grad[self.input_dim * self.output_dim + o] += d;
            }

            for i in 0..self.input_dim {
                let mut sum = 0.0;
                for o in 0..self.output_dim {
                    sum += params[self.weight_index(o, i, slice)] * delta.data[r][o];
                }
                delta_prev[r][i] = sum;
            }
        }
        (Tensor2D::new(delta_prev), grad)
    }

    fn param_len(&self) -> usize {
        self.input_dim * self.output_dim + self.output_dim
    }

    fn in_features(&self) -> usize { self.input_dim }
    fn out_features(&self) -> usize { self.output_dim }
}




