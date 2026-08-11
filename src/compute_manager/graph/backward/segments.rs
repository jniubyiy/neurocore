// src/compute_manager/graph/backward/segments.rs

use faer::Mat;
use crate::compute_manager::dim_change;
use crate::compute_manager::graph::model::MixedModel;

impl MixedModel {
    // ---------- Операции изменения размерности ----------
    pub(super) fn process_unsqueeze_backward(
        &self,
        stream_matrices: &mut Vec<Mat<f32>>,
        target_dims: &[usize],
    ) {
        for mat in stream_matrices.iter_mut() {
            *mat = dim_change::reduce_mat(mat, target_dims);
        }
    }

    pub(super) fn process_reduce_mean_backward(
        &self,
        stream_matrices: &mut Vec<Mat<f32>>,
        target_dims: &[usize],
    ) {
        for mat in stream_matrices.iter_mut() {
            *mat = dim_change::unsqueeze_mat(mat, target_dims);
        }
    }
}