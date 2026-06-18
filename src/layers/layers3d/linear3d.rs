use faer::Mat;
use crate::tensor::Tensor3D;
use crate::model_plan::param_store::ParamSlice;
use crate::linalg;
use super::{Layer3D, LayerContext3D};

pub struct Linear3D {
    pub dim3: usize,
    pub out_dim3: usize,
}

impl Linear3D {
    pub fn new(dim3: usize, out_dim3: usize) -> Self {
        assert!(dim3 > 0 && out_dim3 > 0);
        Self { dim3, out_dim3 }
    }

    fn weight_index(&self, out_idx: usize, in_idx: usize, slice: &ParamSlice) -> usize {
        slice.start + out_idx * self.dim3 + in_idx
    }
}

impl Layer3D for Linear3D {
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
        let w_start = slice.start;
        let w = Mat::from_fn(self.out_dim3, self.dim3, |r, c| params[w_start + r * self.dim3 + c]);
        let b_start = w_start + self.dim3 * self.out_dim3;
        let b = Mat::from_fn(self.out_dim3, 1, |r, _| params[b_start + r]);

        let y = &x * &w.transpose();
        let out_tensor = linalg::faer_to_tensor3d(&y, dim1, dim2, self.out_dim3);
        for i in 0..dim1 {
            for j in 0..dim2 {
                out_buf[i][j].copy_from_slice(&out_tensor.data[i][j]);
            }
        }
        // Добавляем bias
        for i in 0..dim1 {
            for j in 0..dim2 {
                for o in 0..self.out_dim3 {
                    out_buf[i][j][o] += b[(o, 0)];
                }
            }
        }

        vec![LayerContext3D::Linear3D { input: input.clone() }]
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
            LayerContext3D::Linear3D { input } => input,
            _ => panic!("Invalid context for Linear3D"),
        };
        let dim1 = input.dim1;
        let dim2 = input.dim2;
        let dim3 = input.dim3;
        assert_eq!(delta.dim1, dim1);
        assert_eq!(delta.dim2, dim2);
        assert_eq!(delta.dim3, self.out_dim3);

        let mut grad = vec![0.0; self.param_len()];
        let mut delta_prev = vec![vec![vec![0.0; dim3]; dim2]; dim1];

        for i in 0..dim1 {
            for j in 0..dim2 {
                for o in 0..self.out_dim3 {
                    let d = delta.data[i][j][o];
                    for k in 0..dim3 {
                        let idx = o * dim3 + k;
                        grad[idx] += d * input.data[i][j][k];
                    }
                    grad[dim3 * self.out_dim3 + o] += d;
                }
                for k in 0..dim3 {
                    let mut sum = 0.0;
                    for o in 0..self.out_dim3 {
                        sum += params[self.weight_index(o, k, slice)] * delta.data[i][j][o];
                    }
                    delta_prev[i][j][k] = sum;
                }
            }
        }

        (vec![Tensor3D::new(delta_prev)], grad)
    }

    fn param_len(&self) -> usize {
        self.dim3 * self.out_dim3 + self.out_dim3
    }
}





