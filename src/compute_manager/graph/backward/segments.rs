// src/compute_manager/graph/backward/segments.rs

use crate::compute_manager::dim_change::{self, DynamicTensor};
use crate::compute_manager::graph::model::MixedModel;
use crate::linalg;

impl MixedModel {
    // ---------- Операции изменения размерности ----------
    pub(super) fn process_unsqueeze_backward(
        &self,
        streams: &mut Vec<Vec<DynamicTensor>>,
        target_dims: &[usize],
    ) {
        let target_dims = target_dims.to_vec();
        for stream in streams.iter_mut() {
            for d in stream.iter_mut() {
                if let DynamicTensor::Dim1(t) = d {
                    let mat = linalg::tensor2d_to_faer(t);
                    let new_mat = dim_change::reduce_mat(&mat, &target_dims);
                    *d = DynamicTensor::Dim1(linalg::faer_to_tensor2d(&new_mat));
                } else {
                    panic!("Unsqueeze backward requires Dim1 input");
                }
            }
        }
    }

    pub(super) fn process_reduce_mean_backward(
        &self,
        streams: &mut Vec<Vec<DynamicTensor>>,
        target_dims: &[usize],
    ) {
        let target_dims = target_dims.to_vec();
        for stream in streams.iter_mut() {
            for d in stream.iter_mut() {
                if let DynamicTensor::Dim1(t) = d {
                    let mat = linalg::tensor2d_to_faer(t);
                    let new_mat = dim_change::unsqueeze_mat(&mat, &target_dims);
                    *d = DynamicTensor::Dim1(linalg::faer_to_tensor2d(&new_mat));
                } else {
                    panic!("ReduceMean backward requires Dim1 input");
                }
            }
        }
    }
}