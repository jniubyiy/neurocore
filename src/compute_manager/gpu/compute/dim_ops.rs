// src/compute_manager/gpu/compute/dim_ops.rs

use faer::Mat;
use super::base::GpuCompute;
use crate::compute_manager::dim_change;

impl GpuCompute {
    pub fn run_reduce_mat(&self, mat: &Mat<f32>, target_dims: &[usize]) -> Mat<f32> {
        dim_change::reduce_mat(mat, target_dims)
    }

    pub fn run_unsqueeze_mat(&self, mat: &Mat<f32>, target_dims: &[usize]) -> Mat<f32> {
        dim_change::unsqueeze_mat(mat, target_dims)
    }
}