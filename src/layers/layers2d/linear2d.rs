use crate::tensor::Tensor2D;
use crate::jacobian::Jacobian2D;
use crate::model_plan::param_store::ParamSlice;
use crate::model_plan::blueprint::assert_power_of_two;
use super::Layer2D;

pub struct Linear2D {
    pub input_dim: usize,
    pub output_dim: usize,
    slice: ParamSlice,
}

impl Linear2D {
    pub fn new(input_dim: usize, output_dim: usize, slice: ParamSlice) -> Self {
        assert_power_of_two(input_dim);
        assert_power_of_two(output_dim);
        assert_eq!(slice.len, input_dim * output_dim + output_dim);
        Self { input_dim, output_dim, slice }
    }

    fn weight_index(&self, out_idx: usize, in_idx: usize) -> usize {
        self.slice.start + out_idx * self.input_dim + in_idx
    }

    fn bias_index(&self, out_idx: usize) -> usize {
        self.slice.start + self.input_dim * self.output_dim + out_idx
    }
}

impl Layer2D for Linear2D {
    fn forward_2d(
        &self,
        input: &Tensor2D,
        j_input: &Jacobian2D,
        params: &[f32],
        _slice: &ParamSlice,
    ) -> (Tensor2D, Jacobian2D) {
        let rows = input.rows;
        let total_params = j_input.num_params;
        let mut out_data = vec![vec![0.0; self.output_dim]; rows];
        let mut j_out = Jacobian2D::new(rows, self.output_dim, total_params);

        for r in 0..rows {
            for out_i in 0..self.output_dim {
                let mut sum = 0.0;
                for in_i in 0..self.input_dim {
                    let w = params[self.weight_index(out_i, in_i)];
                    sum += w * input.data[r][in_i];
                    let global_idx = self.weight_index(out_i, in_i);
                    if global_idx < total_params {
                        j_out.data[r][out_i][global_idx] += input.data[r][in_i];
                    }
                }
                let b = params[self.bias_index(out_i)];
                sum += b;
                let global_idx = self.bias_index(out_i);
                if global_idx < total_params {
                    j_out.data[r][out_i][global_idx] += 1.0;
                }
                out_data[r][out_i] = sum;

                for p in 0..total_params {
                    let mut deriv = 0.0;
                    for in_i in 0..self.input_dim {
                        let w = params[self.weight_index(out_i, in_i)];
                        deriv += w * j_input.data[r][in_i][p];
                    }
                    j_out.data[r][out_i][p] += deriv;
                }
            }
        }

        (Tensor2D::new(out_data), j_out)
    }

    fn param_len(&self) -> usize { self.slice.len }
    fn in_features(&self) -> usize { self.input_dim }
    fn out_features(&self) -> usize { self.output_dim }

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





