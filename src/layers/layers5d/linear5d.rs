use faer::Mat;
use crate::tensor::Tensor5D;
use crate::model_plan::param_store::ParamSlice;
use crate::linalg;
use super::{Layer5D, LayerContext5D};

pub struct Linear5D {
    pub dim5: usize,       // размер последней оси входа
    pub out_dim5: usize,   // размер последней оси выхода
}

impl Linear5D {
    pub fn new(dim5: usize, out_dim5: usize) -> Self {
        assert!(dim5 > 0 && out_dim5 > 0);
        Self { dim5, out_dim5 }
    }

    fn weight_index(&self, out_idx: usize, in_idx: usize, slice: &ParamSlice) -> usize {
        slice.start + out_idx * self.dim5 + in_idx
    }
}

impl Layer5D for Linear5D {
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
        let w_start = slice.start;
        let w = Mat::from_fn(self.out_dim5, self.dim5, |r, c| params[w_start + r * self.dim5 + c]);
        let b_start = w_start + self.dim5 * self.out_dim5;
        let b = Mat::from_fn(self.out_dim5, 1, |r, _| params[b_start + r]);

        let y = &x * &w.transpose();
        let out_tensor = linalg::faer_to_tensor5d(&y, dim1, dim2, dim3, dim4, self.out_dim5);
        for i in 0..dim1 {
            for j in 0..dim2 {
                for k in 0..dim3 {
                    for l in 0..dim4 {
                        out_buf[i][j][k][l].copy_from_slice(&out_tensor.data[i][j][k][l]);
                    }
                }
            }
        }
        // bias
        for i in 0..dim1 {
            for j in 0..dim2 {
                for k in 0..dim3 {
                    for l in 0..dim4 {
                        for o in 0..self.out_dim5 {
                            out_buf[i][j][k][l][o] += b[(o, 0)];
                        }
                    }
                }
            }
        }

        vec![LayerContext5D::Linear5D { input: input.clone() }]
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
            LayerContext5D::Linear5D { input } => input,
            _ => panic!("Invalid context for Linear5D"),
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

        let mut grad = vec![0.0; self.param_len()];
        let mut delta_prev = vec![vec![vec![vec![vec![0.0; dim5]; dim4]; dim3]; dim2]; dim1];

        for i in 0..dim1 {
            for j in 0..dim2 {
                for k in 0..dim3 {
                    for l in 0..dim4 {
                        for o in 0..self.out_dim5 {
                            let d = delta.data[i][j][k][l][o];
                            for n in 0..dim5 {
                                let idx = o * dim5 + n;
                                grad[idx] += d * input.data[i][j][k][l][n];
                            }
                            grad[dim5 * self.out_dim5 + o] += d;
                        }
                        for n in 0..dim5 {
                            let mut sum = 0.0;
                            for o in 0..self.out_dim5 {
                                sum += params[self.weight_index(o, n, slice)] * delta.data[i][j][k][l][o];
                            }
                            delta_prev[i][j][k][l][n] = sum;
                        }
                    }
                }
            }
        }

        (vec![Tensor5D::new(delta_prev)], grad)
    }

    fn param_len(&self) -> usize {
        self.dim5 * self.out_dim5 + self.out_dim5
    }
}




