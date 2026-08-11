use crate::compute_manager::graph::types::DynamicContext;
use crate::compute_manager::matrix_buffer::MatrixBuffer;
use crate::layers::UniversalLayer;
use crate::layers::UniversalLayerBuffered;
use crate::model_plan::param_store::ParamSlice;
use crate::layers::mat_context::MatContext;
use faer::Mat;

pub struct LeakyReLU {
    pub alpha: f32,
}

impl LeakyReLU {
    pub fn new(alpha: f32) -> Self { Self { alpha } }
}

// ---------------------------------------------------------------------------
// Старая реализация UniversalLayer (оставлена для GPU и обратной совместимости)
// ---------------------------------------------------------------------------

impl UniversalLayer for LeakyReLU {
    fn forward_mat(
        &self,
        input: &Mat<f32>,
        _params: &[f32],
        _slice: &ParamSlice,
    ) -> (Mat<f32>, DynamicContext) {
        let output = input.map(|x| if *x > 0.0 { *x } else { self.alpha * (*x) });
        let ctx = DynamicContext::Mat(MatContext::LeakyReLU { input: input.clone() });
        (output, ctx)
    }

    fn backward_mat(
        &self,
        ctx: &DynamicContext,
        delta: &Mat<f32>,
        _params: &[f32],
        _slice: &ParamSlice,
    ) -> (Mat<f32>, Vec<f32>) {
        let x_mat = match ctx {
            DynamicContext::Mat(MatContext::LeakyReLU { input }) => input.clone(),
            _ => panic!("Expected LeakyReLU context"),
        };
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
        DynamicContext::Mat(MatContext::LeakyReLU { input: input_sample.clone() })
    }

    fn output_mat_shape(&self, _batch_size: usize) -> Mat<f32> {
        Mat::zeros(0, 0)
    }

    fn as_leaky_relu(&self) -> Option<&LeakyReLU> {
        Some(self)
    }
}

// ---------------------------------------------------------------------------
// Новая реализация UniversalLayerBuffered (CPU‑путь с управляемыми буферами)
// ---------------------------------------------------------------------------

impl UniversalLayerBuffered for LeakyReLU {
    fn forward_buffered(
        &self,
        input: &MatrixBuffer,
        output: &mut MatrixBuffer,
        _params: &[f32],
        _slice: &ParamSlice,
    ) {
        let inp = input.as_mat();
        let mut out = output.as_mat_mut();

        // Поэлементно применяем LeakyReLU и записываем результат в выходной буфер
        let temp = Mat::from_fn(inp.nrows(), inp.ncols(), |r, c| {
            let x = inp[(r, c)];
            if x > 0.0 { x } else { self.alpha * x }
        });
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
        let x_mat = match ctx {
            DynamicContext::Mat(MatContext::LeakyReLU { input }) => input.clone(),
            _ => panic!("Expected LeakyReLU context"),
        };
        let go = grad_output.as_mat();
        let mut gi = grad_input.as_mat_mut();

        let dx = Mat::from_fn(x_mat.nrows(), x_mat.ncols(), |r, c| {
            let grad = if x_mat[(r, c)] > 0.0 { 1.0 } else { self.alpha };
            go[(r, c)] * grad
        });
        gi.copy_from(&dx);
        Vec::new()
    }

    fn param_len(&self) -> usize { 0 }
    fn input_features(&self) -> usize { 0 }
    fn output_features(&self) -> usize { 0 }
}