use faer::Mat;
use crate::tensor::Tensor4D;
use crate::model_plan::param_store::ParamSlice;
use crate::linalg;
use super::{Layer4D, LayerContext4D};

pub struct Linear4D {
    pub dim4: usize,
    pub out_dim4: usize,
}

impl Linear4D {
    pub fn new(dim4: usize, out_dim4: usize) -> Self {
        assert!(dim4 > 0 && out_dim4 > 0);
        Self { dim4, out_dim4 }
    }

    fn weight_index(&self, out_idx: usize, in_idx: usize, slice: &ParamSlice) -> usize {
        slice.start + out_idx * self.dim4 + in_idx
    }
}

impl Layer4D for Linear4D {
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
        let w_start = slice.start;
        let w = Mat::from_fn(self.out_dim4, self.dim4, |r, c| params[w_start + r * self.dim4 + c]);
        let b_start = w_start + self.dim4 * self.out_dim4;
        let b = Mat::from_fn(self.out_dim4, 1, |r, _| params[b_start + r]);

        let y = &x * &w.transpose();
        let out_tensor = linalg::faer_to_tensor4d(&y, dim1, dim2, dim3, self.out_dim4);
        for i in 0..dim1 {
            for j in 0..dim2 {
                for k in 0..dim3 {
                    out_buf[i][j][k].copy_from_slice(&out_tensor.data[i][j][k]);
                }
            }
        }
        for i in 0..dim1 {
            for j in 0..dim2 {
                for k in 0..dim3 {
                    for o in 0..self.out_dim4 {
                        out_buf[i][j][k][o] += b[(o, 0)];
                    }
                }
            }
        }

        vec![LayerContext4D::Linear4D { input: input.clone() }]
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
            LayerContext4D::Linear4D { input } => input,
            _ => panic!("Invalid context for Linear4D"),
        };
        let dim1 = input.dim1;
        let dim2 = input.dim2;
        let dim3 = input.dim3;
        let dim4 = input.dim4;
        assert_eq!(delta.dim1, dim1);
        assert_eq!(delta.dim2, dim2);
        assert_eq!(delta.dim3, dim3);
        assert_eq!(delta.dim4, self.out_dim4);

        let mut grad = vec![0.0; self.param_len()];
        let mut delta_prev = vec![vec![vec![vec![0.0; dim4]; dim3]; dim2]; dim1];

        for i in 0..dim1 {
            for j in 0..dim2 {
                for k in 0..dim3 {
                    for o in 0..self.out_dim4 {
                        let d = delta.data[i][j][k][o];
                        for n in 0..dim4 {
                            let idx = o * dim4 + n;
                            grad[idx] += d * input.data[i][j][k][n];
                        }
                        grad[dim4 * self.out_dim4 + o] += d;
                    }
                    for n in 0..dim4 {
                        let mut sum = 0.0;
                        for o in 0..self.out_dim4 {
                            sum += params[self.weight_index(o, n, slice)] * delta.data[i][j][k][o];
                        }
                        delta_prev[i][j][k][n] = sum;
                    }
                }
            }
        }

        (vec![Tensor4D::new(delta_prev)], grad)
    }

    fn param_len(&self) -> usize {
        self.dim4 * self.out_dim4 + self.out_dim4
    }
}





