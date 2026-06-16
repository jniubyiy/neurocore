use crate::tensor::Tensor3D;
use crate::jacobian::Jacobian3D;
use crate::model_plan::param_store::ParamSlice;
use super::Layer3D;

pub struct Softmax3D;

impl Softmax3D {
    pub fn new() -> Self { Self }
}

impl Layer3D for Softmax3D {
    fn forward_3d(
        &self,
        input: &Tensor3D,
        j_input: &Jacobian3D,
        _params: &[f32],
        _slice: &ParamSlice,
    ) -> (Tensor3D, Jacobian3D) {
        let depth = input.depth;
        let rows = input.rows;
        let cols = input.cols;
        let params = j_input.num_params;
        let mut out = vec![vec![vec![0.0; cols]; rows]; depth];
        let mut j_out = Jacobian3D::new(depth, rows, cols, params);
        for d in 0..depth {
            for r in 0..rows {
                let row = &input.data[d][r];
                let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut exps = vec![0.0; cols];
                let mut sum = 0.0;
                for c in 0..cols {
                    exps[c] = (row[c] - max_val).exp();
                    sum += exps[c];
                }
                for c in 0..cols {
                    let softmax_val = exps[c] / sum;
                    out[d][r][c] = softmax_val;
                    for p in 0..params {
                        let mut deriv = 0.0;
                        for k in 0..cols {
                            let delta = if c == k { 1.0 } else { 0.0 };
                            deriv += softmax_val * (delta - exps[k] / sum) * j_input.data[d][r][k][p];
                        }
                        j_out.data[d][r][c][p] = deriv;
                    }
                }
            }
        }
        (Tensor3D::new(out), j_out)
    }

    fn param_len(&self) -> usize { 0 }
}





