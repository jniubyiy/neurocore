use crate::tensor::Tensor3D;
use crate::jacobian::Jacobian3D;
use crate::model_plan::param_store::ParamSlice;
use super::Layer3D;

pub struct ReLU3D;

impl ReLU3D {
    pub fn new() -> Self { Self }
}

impl Layer3D for ReLU3D {
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
                for c in 0..cols {
                    let x = input.data[d][r][c];
                    let activated = if x > 0.0 { x } else { 0.0 };
                    out[d][r][c] = activated;
                    let grad = if x > 0.0 { 1.0 } else { 0.0 };
                    for p in 0..params {
                        j_out.data[d][r][c][p] = j_input.data[d][r][c][p] * grad;
                    }
                }
            }
        }
        (Tensor3D::new(out), j_out)
    }

    fn param_len(&self) -> usize { 0 }
}





