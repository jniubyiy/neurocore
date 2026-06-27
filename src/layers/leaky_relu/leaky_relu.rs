use crate::compute_manager::graph::types::DynamicContext;
use crate::layers::UniversalLayer;
use crate::model_plan::param_store::ParamSlice;
use crate::linalg;
use faer::Mat;

pub struct LeakyReLU {
    pub alpha: f32,
}

impl LeakyReLU {
    pub fn new(alpha: f32) -> Self { Self { alpha } }
}

impl UniversalLayer for LeakyReLU {
    fn forward_mat(
        &self,
        input: &Mat<f32>,
        _params: &[f32],
        _slice: &ParamSlice,
    ) -> (Mat<f32>, DynamicContext) {
        let output = input.map(|x| if *x > 0.0 { *x } else { self.alpha * (*x) });
        let input_tensor = linalg::faer_to_tensor2d(input);
        let ctx = DynamicContext::Ctx1D(
            crate::layers::context1d::LayerContext1D::LeakyReLU { input: input_tensor },
        );
        (output, ctx)
    }

    fn backward_mat(
        &self,
        ctx: &DynamicContext,
        delta: &Mat<f32>,
        _params: &[f32],
        _slice: &ParamSlice,
    ) -> (Mat<f32>, Vec<f32>) {
        let x_tensor = match ctx {
            DynamicContext::Ctx1D(c) => match c {
                crate::layers::context1d::LayerContext1D::LeakyReLU { input } => input,
                _ => panic!("Expected LeakyReLU context"),
            },
            _ => panic!("Expected Ctx1D context"),
        };
        let x_mat = linalg::tensor2d_to_faer(x_tensor);
        let dx = Mat::from_fn(x_mat.nrows(), x_mat.ncols(), |r, c| {
            let grad = if x_mat[(r, c)] > 0.0 { 1.0 } else { self.alpha };
            delta[(r, c)] * grad
        });
        (dx, vec![])
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
        let chunk = input.submatrix(task_offset, 0, task_count, input.ncols());
        let out_chunk = chunk.map(|x| if *x > 0.0 { *x } else { self.alpha * (*x) });
        for r in 0..task_count {
            for c in 0..out_chunk.ncols() {
                output[(task_offset + r, c)] = out_chunk[(r, c)];
            }
        }
    }

    fn create_sample_context(
        &self,
        input_sample: &Mat<f32>,
        _output_sample: &Mat<f32>,
    ) -> DynamicContext {
        let t = linalg::faer_to_tensor2d(input_sample);
        DynamicContext::Ctx1D(
            crate::layers::context1d::LayerContext1D::LeakyReLU { input: t },
        )
    }

    fn output_mat_shape(&self, _batch_size: usize) -> Mat<f32> {
        Mat::zeros(0, 0)
    }

    fn as_leaky_relu(&self) -> Option<&LeakyReLU> {
        Some(self)
    }
}