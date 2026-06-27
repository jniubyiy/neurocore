use crate::compute_manager::graph::types::DynamicContext;
use crate::layers::UniversalLayer;
use crate::model_plan::param_store::ParamSlice;
use crate::linalg;
use faer::Mat;

pub struct Identity;

impl Identity {
    pub fn new() -> Self { Self }
}

impl UniversalLayer for Identity {
    fn forward_mat(
        &self,
        input: &Mat<f32>,
        _params: &[f32],
        _slice: &ParamSlice,
    ) -> (Mat<f32>, DynamicContext) {
        let ctx = DynamicContext::Ctx1D(
            crate::layers::context1d::LayerContext1D::Linear {
                input: linalg::faer_to_tensor2d(input),
            },
        );
        (input.clone(), ctx)
    }

    fn backward_mat(
        &self,
        _ctx: &DynamicContext,
        delta: &Mat<f32>,
        _params: &[f32],
        _slice: &ParamSlice,
    ) -> (Mat<f32>, Vec<f32>) {
        (delta.clone(), vec![])
    }

    fn param_len(&self) -> usize { 0 }
    fn input_features(&self) -> usize { 0 }
    fn output_features(&self) -> usize { 0 }

    fn total_tasks(&self, batch_size: usize) -> usize { batch_size }

    fn execute_tasks(
        &self,
        input: &Mat<f32>,
        output: &mut Mat<f32>,
        task_offset: usize,
        task_count: usize,
        _params: &[f32],
        _slice: &ParamSlice,
    ) {
        let ncols = input.ncols();
        for r in 0..task_count {
            for c in 0..ncols {
                output[(task_offset + r, c)] = input[(task_offset + r, c)];
            }
        }
    }

    fn create_sample_context(
        &self,
        input_sample: &Mat<f32>,
        _output_sample: &Mat<f32>,
    ) -> DynamicContext {
        DynamicContext::Ctx1D(
            crate::layers::context1d::LayerContext1D::Linear {
                input: linalg::faer_to_tensor2d(input_sample),
            },
        )
    }

    fn output_mat_shape(&self, _batch_size: usize) -> Mat<f32> {
        Mat::zeros(0, 0)
    }

    fn as_identity(&self) -> Option<&Identity> {
        Some(self)
    }
}