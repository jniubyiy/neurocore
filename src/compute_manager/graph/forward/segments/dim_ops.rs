// src/compute_manager/graph/forward/segments/dim_ops.rs

use crate::compute_manager::dim_change::{self, DynamicTensor};
use crate::compute_manager::graph::model::MixedModel;

impl MixedModel {
    /// Прямой проход Unsqueeze (reshape вверх) с заданными целевыми размерами.
    pub(crate) fn process_unsqueeze_forward(
        &self,
        streams: &mut Vec<Vec<DynamicTensor>>,
        target_dims: &[usize],
    ) {
        let target_dims = target_dims.to_vec();
        for stream in streams.iter_mut() {
            for sample in stream.iter_mut() {
                *sample = dim_change::unsqueeze_to(sample.clone(), target_dims.clone());
            }
        }
    }

    /// Прямой проход ReduceMean (reshape вниз) с заданными целевыми размерами.
    pub(crate) fn process_reduce_mean_forward(
        &self,
        streams: &mut Vec<Vec<DynamicTensor>>,
        target_dims: &[usize],
    ) {
        let target_dims = target_dims.to_vec();
        for stream in streams.iter_mut() {
            for sample in stream.iter_mut() {
                *sample = dim_change::reduce_to(sample.clone(), target_dims.clone());
            }
        }
    }
}