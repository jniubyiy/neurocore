// src/compute_manager/gpu/compute/linear.rs

use faer::Mat;
use super::base::GpuCompute;
use crate::model_plan::param_store::ParamSlice;

impl GpuCompute {
    pub fn run_linear_forward(
        &self,
        input: &Mat<f32>,
        weight: &Mat<f32>,
        bias: &[f32],
    ) -> Mat<f32> {
        let weight_t = Mat::from_fn(weight.ncols(), weight.nrows(), |r, c| weight[(c, r)]);
        let mut out = self.run_mat_mul(input, &weight_t);
        let batch = input.nrows();
        let out_features = weight.nrows();
        for r in 0..batch {
            for c in 0..out_features {
                out[(r, c)] += bias[c];
            }
        }
        out
    }

    pub fn run_linear_backward(
        &self,
        input: &Mat<f32>,
        weight: &Mat<f32>,
        grad_output: &Mat<f32>,
    ) -> (Mat<f32>, Mat<f32>, Vec<f32>) {
        let grad_input = self.run_mat_mul(grad_output, weight);
        let grad_output_t = Mat::from_fn(grad_output.ncols(), grad_output.nrows(), |r, c| grad_output[(c, r)]);
        let grad_weight = self.run_mat_mul(&grad_output_t, input);
        let grad_bias = self.run_reduce_sum_cols(grad_output);
        (grad_input, grad_weight, grad_bias)
    }
}