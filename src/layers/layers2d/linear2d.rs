// src/layers/layers2d/linear2d.rs (удалены assert_power_of_two)

use faer::Mat;
use crate::tensor::Tensor2D;
use crate::model_plan::param_store::ParamSlice;
use crate::linalg;
use super::{Layer2D, LayerContext};

pub struct Linear2D {
    pub dim2: usize,
    pub out_dim2: usize,
}

impl Linear2D {
    pub fn new(dim2: usize, out_dim2: usize) -> Self {
        // Проверки степеней двойки удалены
        Self { dim2, out_dim2 }
    }

    fn weight_index(&self, out_idx: usize, in_idx: usize, slice: &ParamSlice) -> usize {
        slice.start + out_idx * self.dim2 + in_idx
    }
}

impl Layer2D for Linear2D {
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

        let w_start = slice.start;
        let w = Mat::from_fn(self.out_dim2, self.dim2, |r, c| params[w_start + r * self.dim2 + c]);
        let b_start = w_start + self.dim2 * self.out_dim2;
        let b = Mat::from_fn(self.out_dim2, 1, |r, _| params[b_start + r]);

        let y = &x * &w.transpose();

        for r in 0..dim1 {
            for c in 0..self.out_dim2 {
                out_buf[r][c] = y[(r, c)] + b[(c, 0)];
            }
        }

        vec![LayerContext::Linear2D { input: input.clone() }]
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
            LayerContext::Linear2D { input } => input,
            _ => panic!("Invalid context for Linear2D"),
        };
        let dim1 = input.dim1;
        assert_eq!(delta.dim1, dim1);
        assert_eq!(delta.dim2, self.out_dim2);

        let mut grad = vec![0.0; self.param_len()];
        let mut delta_prev = vec![vec![0.0; self.dim2]; dim1];

        for r in 0..dim1 {
            for o in 0..self.out_dim2 {
                let d = delta.data[r][o];
                for i in 0..self.dim2 {
                    let idx = o * self.dim2 + i;
                    grad[idx] += d * input.data[r][i];
                }
                grad[self.dim2 * self.out_dim2 + o] += d;
            }

            for i in 0..self.dim2 {
                let mut sum = 0.0;
                for o in 0..self.out_dim2 {
                    sum += params[self.weight_index(o, i, slice)] * delta.data[r][o];
                }
                delta_prev[r][i] = sum;
            }
        }

        (vec![Tensor2D::new(delta_prev)], grad)
    }

    fn param_len(&self) -> usize {
        self.dim2 * self.out_dim2 + self.out_dim2
    }
}




