// src/compute_manager/graph/backward/segments.rs

use crate::compute_manager::dim_change::DynamicTensor;
use crate::compute_manager::graph::model::MixedModel;

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
                *d = crate::compute_manager::dim_change::reduce_to(d.clone(), target_dims.clone());
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
                *d = crate::compute_manager::dim_change::unsqueeze_to(d.clone(), target_dims.clone());
            }
        }
    }
}