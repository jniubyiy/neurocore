use crate::compute_manager::graph::types::DynamicContext;
use crate::layers::UniversalLayer;
use crate::model_plan::param_store::ParamSlice;
use crate::linalg;
use faer::Mat;

pub struct Tanh;

impl Tanh {
    pub fn new() -> Self { Self }
}

impl UniversalLayer for Tanh {
    fn forward_mat(
        &self,
        input: &Mat<f32>,
        _params: &[f32],
        _slice: &ParamSlice,
    ) -> (Mat<f32>, DynamicContext) {
        let output = input.map(|x| x.tanh());
        let output_tensor = linalg::faer_to_tensor2d(&output);
        let ctx = DynamicContext::Ctx1D(
            crate::layers::context1d::LayerContext1D::Tanh { output: output_tensor },
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
        let y_tensor = match ctx {
            DynamicContext::Ctx1D(c) => match c {
                crate::layers::context1d::LayerContext1D::Tanh { output } => output,
                _ => panic!("Expected Tanh context"),
            },
            _ => panic!("Expected Ctx1D context"),
        };
        let y_mat = linalg::tensor2d_to_faer(y_tensor);
        let dx = Mat::from_fn(y_mat.nrows(), y_mat.ncols(), |r, c| {
            let val = y_mat[(r, c)];
            delta[(r, c)] * (1.0 - val * val)
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
        let out_chunk = chunk.map(|x| x.tanh());
        for r in 0..task_count {
            for c in 0..out_chunk.ncols() {
                output[(task_offset + r, c)] = out_chunk[(r, c)];
            }
        }
    }

    fn create_sample_context(
        &self,
        _input_sample: &Mat<f32>,
        output_sample: &Mat<f32>,
    ) -> DynamicContext {
        let t = linalg::faer_to_tensor2d(output_sample);
        DynamicContext::Ctx1D(
            crate::layers::context1d::LayerContext1D::Tanh { output: t },
        )
    }

    fn output_mat_shape(&self, _batch_size: usize) -> Mat<f32> {
        Mat::zeros(0, 0)
    }

    fn as_tanh(&self) -> Option<&Tanh> {
        Some(self)
    }
}