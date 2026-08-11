use crate::compute_manager::graph::types::DynamicContext;
use crate::compute_manager::matrix_buffer::MatrixBuffer;
use crate::layers::UniversalLayer;
use crate::layers::UniversalLayerBuffered;
use crate::model_plan::param_store::ParamSlice;
use crate::layers::mat_context::MatContext;
use faer::Mat;

pub struct Softmax;

impl Softmax {
    pub fn new() -> Self { Self }
}

// ---------------------------------------------------------------------------
// Старая реализация UniversalLayer (оставлена для GPU и обратной совместимости)
// ---------------------------------------------------------------------------

impl UniversalLayer for Softmax {
    fn forward_mat(
        &self,
        input: &Mat<f32>,
        _params: &[f32],
        _slice: &ParamSlice,
    ) -> (Mat<f32>, DynamicContext) {
        let output = softmax_forward_mat(input);
        let ctx = DynamicContext::Mat(MatContext::Softmax { output: output.clone() });
        (output, ctx)
    }

    fn backward_mat(
        &self,
        ctx: &DynamicContext,
        delta: &Mat<f32>,
        _params: &[f32],
        _slice: &ParamSlice,
    ) -> (Mat<f32>, Vec<f32>) {
        let y_mat = match ctx {
            DynamicContext::Mat(MatContext::Softmax { output }) => output.clone(),
            _ => panic!("Expected Softmax context"),
        };
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
        DynamicContext::Mat(MatContext::Softmax { output: output_sample.clone() })
    }

    fn output_mat_shape(&self, _batch_size: usize) -> Mat<f32> {
        Mat::zeros(0, 0)
    }

    fn as_softmax(&self) -> Option<&Softmax> {
        Some(self)
    }
}

// ---------------------------------------------------------------------------
// Новая реализация UniversalLayerBuffered (CPU‑путь с управляемыми буферами)
// ---------------------------------------------------------------------------

impl UniversalLayerBuffered for Softmax {
    fn forward_buffered(
        &self,
        input: &MatrixBuffer,
        output: &mut MatrixBuffer,
        _params: &[f32],
        _slice: &ParamSlice,
    ) {
        let inp = input.as_mat();
        let mut out = output.as_mat_mut();

        // Преобразуем MatRef в Mat, чтобы передать в существующую функцию
        let inp_owned = inp.to_owned();
        let temp = softmax_forward_mat_inner(&inp_owned);
        out.copy_from(&temp);
    }

    fn backward_buffered(
        &self,
        ctx: &DynamicContext,
        grad_output: &MatrixBuffer,
        grad_input: &mut MatrixBuffer,
        _params: &[f32],
        _slice: &ParamSlice,
    ) -> Vec<f32> {
        let y_mat = match ctx {
            DynamicContext::Mat(MatContext::Softmax { output }) => output.clone(),
            _ => panic!("Expected Softmax context"),
        };
        let go = grad_output.as_mat();
        let go_owned = go.to_owned();
        let mut gi = grad_input.as_mat_mut();

        let dx = softmax_backward_mat(&y_mat, &go_owned);
        gi.copy_from(&dx);
        Vec::new()
    }

    fn param_len(&self) -> usize { 0 }
    fn input_features(&self) -> usize { 0 }
    fn output_features(&self) -> usize { 0 }
}

// Вспомогательные матричные функции (общие для обеих реализаций)

fn softmax_forward_mat(input: &Mat<f32>) -> Mat<f32> {
    softmax_forward_mat_inner(input)
}

fn softmax_forward_mat_inner(input: &Mat<f32>) -> Mat<f32> {
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