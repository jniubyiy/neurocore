use crate::tensor::Tensor2D;
use crate::jacobian::Jacobian2D;
use crate::model_plan::param_store::ParamSlice;
use crate::model_plan::blueprint::assert_power_of_two;
use super::Layer2D;

pub struct Softmax2D {
    size: usize,
}

impl Softmax2D {
    pub fn new(size: usize) -> Self {
        assert_power_of_two(size);
        Self { size }
    }
}

impl Layer2D for Softmax2D {
    fn forward_2d(
        &self,
        input: &Tensor2D,
        j_input: &Jacobian2D,
        _params: &[f32],
        _slice: &ParamSlice,
    ) -> (Tensor2D, Jacobian2D) {
        let rows = input.rows;
        let cols = input.cols;
        let params = j_input.num_params;
        let mut out = vec![vec![0.0; cols]; rows];
        let mut j_out = Jacobian2D::new(rows, cols, params);
        for r in 0..rows {
            let row_vals = &input.data[r];
            let max_val = row_vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut exps = vec![0.0; cols];
            let mut sum = 0.0;
            for c in 0..cols {
                exps[c] = (row_vals[c] - max_val).exp();
                sum += exps[c];
            }
            for c in 0..cols {
                let sm = exps[c] / sum;
                out[r][c] = sm;
                for p in 0..params {
                    let mut deriv = 0.0;
                    for k in 0..cols {
                        let delta = if c == k { 1.0 } else { 0.0 };
                        deriv += sm * (delta - exps[k] / sum) * j_input.data[r][k][p];
                    }
                    j_out.data[r][c][p] = deriv;
                }
            }
        }
        (Tensor2D::new(out), j_out)
    }

    fn param_len(&self) -> usize { 0 }
    fn in_features(&self) -> usize { self.size }
    fn out_features(&self) -> usize { self.size }

    fn execute_range(
        &self,
        _input: &Tensor2D,
        _j_input: &Jacobian2D,
        _out: &mut [f32],
        _j_out: &mut [f32],
        _row_start: usize,
        _row_end: usize,
        _col_start: usize,
        _col_end: usize,
        _total_params: usize,
        _params: &[f32],
        _slice: &ParamSlice,
    ) {
        unimplemented!()
    }
}




