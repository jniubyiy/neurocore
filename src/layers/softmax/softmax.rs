use crate::compute_manager::graph::types::DynamicContext;
use crate::layers::UniversalLayer;
use crate::model_plan::param_store::ParamSlice;
use crate::linalg;
use faer::Mat;

pub struct Softmax;

impl Softmax {
    pub fn new() -> Self { Self }
}

impl UniversalLayer for Softmax {
    fn forward_mat(
        &self,
        input: &Mat<f32>,
        _params: &[f32],
        _slice: &ParamSlice,
    ) -> (Mat<f32>, DynamicContext) {
        let output = softmax_forward_mat(input);
        let output_tensor = linalg::faer_to_tensor2d(&output);
        let ctx = DynamicContext::Ctx1D(
            crate::layers::context1d::LayerContext1D::Softmax { output: output_tensor },
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
                crate::layers::context1d::LayerContext1D::Softmax { output } => output,
                _ => panic!("Expected Softmax context"),
            },
            _ => panic!("Expected Ctx1D context"),
        };
        let y_mat = linalg::tensor2d_to_faer(y_tensor);
        let dx = softmax_backward_mat(&y_mat, delta);
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
        let out_chunk = softmax_forward_mat(&chunk.to_owned());
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
            crate::layers::context1d::LayerContext1D::Softmax { output: t },
        )
    }

    fn output_mat_shape(&self, _batch_size: usize) -> Mat<f32> {
        Mat::zeros(0, 0) // форма определяется входом
    }
}

// Вспомогательные матричные функции
fn softmax_forward_mat(input: &Mat<f32>) -> Mat<f32> {
    let (batch, n) = (input.nrows(), input.ncols());
    Mat::from_fn(batch, n, |i, j| {
        let row = input.row(i);
        let max_val = (0..n).fold(f32::NEG_INFINITY, |acc, c| acc.max(row[c]));
        let sum: f32 = (0..n).map(|c| (row[c] - max_val).exp()).sum();
        (row[j] - max_val).exp() / sum
    })
}

fn softmax_backward_mat(y: &Mat<f32>, dout: &Mat<f32>) -> Mat<f32> {
    let (batch, n) = (y.nrows(), y.ncols());
    let mut dx = Mat::zeros(batch, n);
    for r in 0..batch {
        let mut dot = 0.0f32;
        for c in 0..n {
            dot += y[(r, c)] * dout[(r, c)];
        }
        for c in 0..n {
            dx[(r, c)] = y[(r, c)] * (dout[(r, c)] - dot);
        }
    }
    dx
}