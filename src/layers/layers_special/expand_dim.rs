// src/layers/layers_special/expand_dim.rs

use crate::compute_manager::dim_change::{self, DynamicTensor};
use crate::compute_manager::graph::types::DynamicContext;
use crate::layers::UniversalLayer;
use crate::model_plan::param_store::ParamSlice;
use crate::tensor::Tensor2D;
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
        let tensor = crate::linalg::faer_to_tensor2d(input);
        let dyn_in = DynamicTensor::Dim1(tensor);
        let dyn_out = dim_change::unsqueeze_to(dyn_in, self.target_dims.clone());
        let out_tensor = match &dyn_out {
            DynamicTensor::Dim1(t) => t.clone(),
            _ => panic!("Expected Dim1 after unsqueeze"),
        };
        let out_mat = crate::linalg::tensor2d_to_faer(&out_tensor);
        let ctx = DynamicContext::Ctx1D(
            crate::layers::context1d::LayerContext1D::Linear {
                input: Tensor2D::zeros(1, 0),
            },
        );
        (out_mat, ctx)
    }

    fn backward_mat(
        &self,
        _ctx: &DynamicContext,
        delta: &Mat<f32>,
        _params: &[f32],
        _slice: &ParamSlice,
    ) -> (Mat<f32>, Vec<f32>) {
        let dyn_delta = DynamicTensor::Dim1(crate::linalg::faer_to_tensor2d(delta));
        let dyn_in = dim_change::reduce_to(dyn_delta, self.target_dims.clone());
        let in_tensor = match &dyn_in {
            DynamicTensor::Dim1(t) => t.clone(),
            _ => panic!("Expected Dim1 after reduce"),
        };
        let dx = crate::linalg::tensor2d_to_faer(&in_tensor);
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
        _input_sample: &Mat<f32>,
        _output_sample: &Mat<f32>,
    ) -> DynamicContext {
        DynamicContext::Ctx1D(
            crate::layers::context1d::LayerContext1D::Linear {
                input: Tensor2D::zeros(1, 0),
            },
        )
    }

    fn output_mat_shape(&self, _batch_size: usize) -> Mat<f32> {
        Mat::zeros(0, 0) // форма определяется входом
    }
}





