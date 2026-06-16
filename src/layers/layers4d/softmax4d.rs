use crate::tensor::Tensor4D;
use crate::jacobian::Jacobian4D;
use crate::model_plan::param_store::ParamSlice;
use super::Layer4D;

pub struct Softmax4D;

impl Softmax4D {
    pub fn new() -> Self { Self }
}

impl Layer4D for Softmax4D {
    fn forward_4d(
        &self,
        input: &Tensor4D,
        j_input: &Jacobian4D,
        _params: &[f32],
        _slice: &ParamSlice,
    ) -> (Tensor4D, Jacobian4D) {
        let dim1 = input.dim1;
        let depth = input.depth;
        let rows = input.rows;
        let cols = input.cols;
        let params = j_input.num_params;
        let mut out = vec![vec![vec![vec![0.0; cols]; rows]; depth]; dim1];
        let mut j_out = Jacobian4D::new(dim1, depth, rows, cols, params);
        for d1 in 0..dim1 {
            for d in 0..depth {
                for r in 0..rows {
                    let row_vals = &input.data[d1][d][r];
                    let max_val = row_vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let mut exps = vec![0.0; cols];
                    let mut sum = 0.0;
                    for c in 0..cols {
                        exps[c] = (row_vals[c] - max_val).exp();
                        sum += exps[c];
                    }
                    for c in 0..cols {
                        let sm = exps[c] / sum;
                        out[d1][d][r][c] = sm;
                        for p in 0..params {
                            let mut deriv = 0.0;
                            for k in 0..cols {
                                let delta = if c == k { 1.0 } else { 0.0 };
                                deriv += sm * (delta - exps[k] / sum) * j_input.data[d1][d][r][k][p];
                            }
                            j_out.data[d1][d][r][c][p] = deriv;
                        }
                    }
                }
            }
        }
        (Tensor4D::new(out), j_out)
    }

    fn param_len(&self) -> usize { 0 }
}



