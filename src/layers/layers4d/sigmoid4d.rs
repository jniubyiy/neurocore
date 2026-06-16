use crate::tensor::Tensor4D;
use crate::jacobian::Jacobian4D;
use crate::model_plan::param_store::ParamSlice;
use super::Layer4D;

pub struct Sigmoid4D;

impl Sigmoid4D {
    pub fn new() -> Self { Self }
}

impl Layer4D for Sigmoid4D {
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
                    for c in 0..cols {
                        let x = input.data[d1][d][r][c];
                        let sig = 1.0 / (1.0 + (-x).exp());
                        out[d1][d][r][c] = sig;
                        let grad = sig * (1.0 - sig);
                        for p in 0..params {
                            j_out.data[d1][d][r][c][p] = j_input.data[d1][d][r][c][p] * grad;
                        }
                    }
                }
            }
        }
        (Tensor4D::new(out), j_out)
    }

    fn param_len(&self) -> usize { 0 }
}





