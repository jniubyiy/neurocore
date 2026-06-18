use faer::Mat;
use crate::tensor::Tensor1D;
use crate::model_plan::param_store::ParamSlice;
use crate::linalg;
use super::{Layer, LayerContext1D, LayerInfo};

pub struct LinearLayer {
    pub dim1: usize,
    pub out_dim1: usize,
}

impl LinearLayer {
    pub fn new(dim1: usize, out_dim1: usize) -> Self {
        Self { dim1, out_dim1 }
    }
}

impl Layer for LinearLayer {
    fn input_dim1s(&self) -> Vec<usize> {
        vec![self.dim1]
    }

    fn output_dim1s(&self) -> Vec<usize> {
        vec![self.out_dim1]
    }

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

        let x = linalg::tensor1d_to_faer(input);

        let w_start = slice.start;
        let w = Mat::from_fn(self.out_dim1, self.dim1, |r, c| params[w_start + r * self.dim1 + c]);
        let b_start = w_start + self.dim1 * self.out_dim1;
        let b = Mat::from_fn(self.out_dim1, 1, |r, _| params[b_start + r]);

        let out = &x * &w.transpose();
        for j in 0..self.out_dim1 {
            out_buf[j] = out[(0, j)] + b[(j, 0)];
        }

        vec![LayerContext1D::Linear { input: input.clone() }]
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
            LayerContext1D::Linear { input } => input,
            _ => panic!("Invalid context for Linear1D"),
        };
        assert_eq!(delta.dim1(), self.out_dim1);

        let mut grad = vec![0.0; self.param_len()];
        let mut delta_prev = vec![0.0; self.dim1];

        for o in 0..self.out_dim1 {
            let d = delta.data[o];
            for i in 0..self.dim1 {
                let idx = o * self.dim1 + i;
                grad[idx] += d * input.data[i];
            }
            grad[self.dim1 * self.out_dim1 + o] += d;
        }

        for i in 0..self.dim1 {
            let mut sum = 0.0;
            for o in 0..self.out_dim1 {
                sum += params[slice.start + o * self.dim1 + i] * delta.data[o];
            }
            delta_prev[i] = sum;
        }

        (vec![Tensor1D::new(delta_prev)], grad)
    }

    fn param_len(&self) -> usize {
        self.dim1 * self.out_dim1 + self.out_dim1
    }

    fn layer_info(&self) -> LayerInfo {
        LayerInfo {
            layer_type: "Linear".to_string(),
            input_dim1s: self.input_dim1s(),
            output_dim1s: self.output_dim1s(),
            param_count: self.param_len(),
            param_start_index: None,
        }
    }
}



