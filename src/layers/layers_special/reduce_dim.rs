// src/layers/layers_special/reduce_dim.rs

use crate::compute_manager::dim_change;
use crate::compute_manager::graph::types::DynamicContext;
use crate::layers::UniversalLayer;
use crate::layers::mat_context::MatContext;
use crate::model_plan::param_store::ParamSlice;
use faer::Mat;

pub struct ReduceMean {
    pub input_dims: Vec<usize>,
    pub target_dims: Vec<usize>,
}

impl ReduceMean {
    pub fn with_dims(input_dims: Vec<usize>, target_dims: Vec<usize>) -> Self {
        assert_eq!(input_dims.len(), target_dims.len() + 1,
            "ReduceMean: target_dims must have exactly one less dimension than input_dims");
        let input_total: usize = input_dims.iter().product();
        let target_total: usize = target_dims.iter().product();
        assert_eq!(input_total, target_total,
            "ReduceMean: total number of elements must be conserved");
        Self { input_dims, target_dims }
    }

    pub fn with_target_dims(target_dims: Vec<usize>) -> Self {
        Self { input_dims: Vec::new(), target_dims }
    }
}

impl UniversalLayer for ReduceMean {
    fn forward_mat(
        &self,
        input: &Mat<f32>,
        _params: &[f32],
        _slice: &ParamSlice,
    ) -> (Mat<f32>, DynamicContext) {
        let out_mat = dim_change::reduce_mat(input, &self.target_dims);
        let ctx = DynamicContext::Mat(MatContext::ReduceMean {
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
        let dx = dim_change::unsqueeze_mat(delta, &self.input_dims);
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
        DynamicContext::Mat(MatContext::ReduceMean {
            input: input_sample.clone(),
        })
    }

    fn output_mat_shape(&self, _batch_size: usize) -> Mat<f32> {
        Mat::zeros(0, 0)
    }

    fn as_reduce_mean(&self) -> Option<&ReduceMean> {
        Some(self)
    }
}





