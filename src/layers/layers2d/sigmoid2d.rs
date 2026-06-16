use crate::tensor::Tensor2D;
use crate::jacobian::Jacobian2D;
use crate::model_plan::param_store::ParamSlice;
use crate::model_plan::blueprint::assert_power_of_two;
use super::Layer2D;

pub struct Sigmoid2D {
    size: usize,
}

impl Sigmoid2D {
    pub fn new(size: usize) -> Self {
        assert_power_of_two(size);
        Self { size }
    }
}

impl Layer2D for Sigmoid2D {
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
            for c in 0..cols {
                let x = input.data[r][c];
                let sig = 1.0 / (1.0 + (-x).exp());
                out[r][c] = sig;
                let grad = sig * (1.0 - sig);
                for p in 0..params {
                    j_out.data[r][c][p] = j_input.data[r][c][p] * grad;
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
