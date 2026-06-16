use crate::tensor::Tensor2D;
use crate::jacobian::Jacobian2D;
use crate::model_plan::param_store::ParamSlice;
use crate::model_plan::blueprint::assert_power_of_two;
use super::Layer2D;

pub struct Memory2D {
    pub input_dim: usize,
    pub output_dim: usize,
    slice: ParamSlice,
}

impl Memory2D {
    pub fn new(input_dim: usize, output_dim: usize, slice: ParamSlice) -> Self {
        assert_power_of_two(input_dim);
        assert_power_of_two(output_dim);
        assert_eq!(slice.len, output_dim * (2 * input_dim + 1));
        Self { input_dim, output_dim, slice }
    }
}

impl Layer2D for Memory2D {
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
                let offset = self.slice.start + out_i * (2 * self.input_dim + 1);
                let m0: Vec<f32> = (0..self.input_dim).map(|i| params[offset + i]).collect();
                let m1: Vec<f32> = (0..self.input_dim).map(|i| params[offset + self.input_dim + i]).collect();
                let t_val = params[offset + 2 * self.input_dim];

                let mut dot0 = 0.0;
                let mut dot1 = 0.0;
                for i in 0..self.input_dim {
                    dot0 += input.data[r][i] * m0[i];
                    dot1 += input.data[r][i] * m1[i];
                }

                let logit0 = dot0 / t_val;
                let logit1 = dot1 / t_val;
                let max_logit = logit0.max(logit1);
                let exp0 = (logit0 - max_logit).exp();
                let exp1 = (logit1 - max_logit).exp();
                let sum_exp = exp0 + exp1;
                let soft0 = exp0 / sum_exp;
                let soft1 = exp1 / sum_exp;

                let y_val = soft0 * dot0 + soft1 * dot1;
                out_data[r][out_i] = y_val;

                let ds0_dot0 = soft0 * (1.0 - soft0) / t_val;
                let ds0_dot1 = -soft0 * soft1 / t_val;
                let ds1_dot0 = -soft1 * soft0 / t_val;
                let ds1_dot1 = soft1 * (1.0 - soft1) / t_val;

                let dy_dot0 = soft0 + dot0 * ds0_dot0 + dot1 * ds1_dot0;
                let dy_dot1 = soft1 + dot0 * ds0_dot1 + dot1 * ds1_dot1;

                for p in 0..total_params {
                    let mut grad = 0.0;
                    for k in 0..self.input_dim {
                        let dy_dxk = dy_dot0 * m0[k] + dy_dot1 * m1[k];
                        grad += dy_dxk * j_input.data[r][k][p];
                    }
                    for k in 0..self.input_dim {
                        let idx = offset + k;
                        if idx < total_params && p == idx {
                            grad += dy_dot0 * input.data[r][k];
                        }
                    }
                    for k in 0..self.input_dim {
                        let idx = offset + self.input_dim + k;
                        if idx < total_params && p == idx {
                            grad += dy_dot1 * input.data[r][k];
                        }
                    }
                    let avg_dot = soft0 * dot0 + soft1 * dot1;
                    let ds0_dt = soft0 * (dot0 - avg_dot) / (t_val * t_val);
                    let ds1_dt = soft1 * (dot1 - avg_dot) / (t_val * t_val);
                    let dy_dt = dot0 * ds0_dt + dot1 * ds1_dt;
                    let idx_t = offset + 2 * self.input_dim;
                    if idx_t < total_params && p == idx_t {
                        grad += dy_dt;
                    }
                    j_out.data[r][out_i][p] += grad;
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