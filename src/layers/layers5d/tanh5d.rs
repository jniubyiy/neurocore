use crate::tensor::Tensor5D;
use crate::jacobian::Jacobian5D;
use crate::model_plan::param_store::ParamSlice;
use super::Layer5D;

pub struct Tanh5D;

impl Tanh5D {
    pub fn new() -> Self { Self }
}

impl Layer5D for Tanh5D {
    fn forward_5d(
        &self,
        input: &Tensor5D,
        j_input: &Jacobian5D,
        _params: &[f32],
        _slice: &ParamSlice,
    ) -> (Tensor5D, Jacobian5D) {
        let outer = input.outer;
        let dim1 = input.dim1;
        let depth = input.depth;
        let rows = input.rows;
        let cols = input.cols;
        let params = j_input.num_params;
        let mut out = vec![vec![vec![vec![vec![0.0; cols]; rows]; depth]; dim1]; outer];
        let mut j_out = Jacobian5D::new(outer, dim1, depth, rows, cols, params);
        for o in 0..outer {
            for d1 in 0..dim1 {
                for d in 0..depth {
                    for r in 0..rows {
                        for c in 0..cols {
                            let x = input.data[o][d1][d][r][c];
                            let t = x.tanh();
                            out[o][d1][d][r][c] = t;
                            let grad = 1.0 - t * t;
                            for p in 0..params {
                                j_out.data[o][d1][d][r][c][p] = j_input.data[o][d1][d][r][c][p] * grad;
                            }
                        }
                    }
                }
            }
        }
        (Tensor5D::new(out), j_out)
    }

    fn param_len(&self) -> usize { 0 }
}