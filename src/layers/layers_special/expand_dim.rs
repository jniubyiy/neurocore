// src/layers/layers_special/expand_dim.rs

use crate::compute_manager::dim_change;
use crate::compute_manager::graph::types::DynamicContext;
use crate::layers::UniversalLayer;
use crate::layers::mat_context::MatContext;
use crate::model_plan::param_store::ParamSlice;
use faer::Mat;

pub struct Unsqueeze {
    pub target_dims: Vec<usize>,
}

impl Unsqueeze {
    pub fn with_target_dims(target_dims: Vec<usize>) -> Self {
        Self { target_dims }
    }
}

impl UniversalLayer for Unsqueeze {
    fn forward_mat(
        &self,
        input: &Mat<f32>,
        _params: &[f32],
        _slice: &ParamSlice,
    ) -> (Mat<f32>, DynamicContext) {
        let out_mat = dim_change::unsqueeze_mat(input, &self.target_dims);
        let ctx = DynamicContext::Mat(MatContext::Unsqueeze {
            input: input.clone(),
        });
        (out_mat, ctx)
    }

    fn backward_mat(
        &self,
        _ctx: &DynamicContext,
        delta: &Mat<f32>,
        _params: &[f32],
        _slice: &ParamSlice,
    ) -> (Mat<f32>, Vec<f32>) {
        let dx = dim_change::reduce_mat(delta, &self.target_dims);
        (dx, vec![])
    }

    fn param_len(&self) -> usize { 0 }
    fn input_features(&self) -> usize { 0 }
    fn output_features(&self) -> usize { 0 }

    fn total_tasks(&self, _batch_size: usize) -> usize { 0 }

    fn execute_tasks(
        &self,
        _input: &Mat<f32>,
        _output: &mut Mat<f32>,
        _task_offset: usize,
        _task_count: usize,
        _params: &[f32],
        _slice: &ParamSlice,
    ) {}

    fn create_sample_context(
        &self,
        input_sample: &Mat<f32>,
        _output_sample: &Mat<f32>,
    ) -> DynamicContext {
        DynamicContext::Mat(MatContext::Unsqueeze {
            input: input_sample.clone(),
        })
    }

    fn output_mat_shape(&self, _batch_size: usize) -> Mat<f32> {
        Mat::zeros(0, 0) // форма определяется входом
    }

    fn as_unsqueeze(&self) -> Option<&Unsqueeze> {
        Some(self)
    }
}





