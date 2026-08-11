use crate::compute_manager::graph::types::DynamicContext;
use crate::compute_manager::matrix_buffer::MatrixBuffer;
use crate::layers::UniversalLayer;
use crate::layers::UniversalLayerBuffered;
use crate::model_plan::param_store::ParamSlice;
use crate::layers::mat_context::MatContext;
use faer::Mat;

pub struct Tanh;

impl Tanh {
    pub fn new() -> Self { Self }
}

// ---------------------------------------------------------------------------
// Старая реализация UniversalLayer (оставлена для GPU и обратной совместимости)
// ---------------------------------------------------------------------------

impl UniversalLayer for Tanh {
    fn forward_mat(
        &self,
        input: &Mat<f32>,
        _params: &[f32],
        _slice: &ParamSlice,
    ) -> (Mat<f32>, DynamicContext) {
        let output = input.map(|x| x.tanh());
        let ctx = DynamicContext::Mat(MatContext::Tanh { output: output.clone() });
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
            DynamicContext::Mat(MatContext::Tanh { output }) => output.clone(),
            _ => panic!("Expected Tanh context"),
        };
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
        DynamicContext::Mat(MatContext::Tanh { output: output_sample.clone() })
    }

    fn output_mat_shape(&self, _batch_size: usize) -> Mat<f32> {
        Mat::zeros(0, 0)
    }

    fn as_tanh(&self) -> Option<&Tanh> {
        Some(self)
    }
}

// ---------------------------------------------------------------------------
// Новая реализация UniversalLayerBuffered (CPU‑путь с управляемыми буферами)
// ---------------------------------------------------------------------------

impl UniversalLayerBuffered for Tanh {
    fn forward_buffered(
        &self,
        input: &MatrixBuffer,
        output: &mut MatrixBuffer,
        _params: &[f32],
        _slice: &ParamSlice,
    ) {
        let inp = input.as_mat();
        let mut out = output.as_mat_mut();

        // Вычисляем tanh во временную матрицу и копируем в выходной буфер
        let temp = inp.map(|x| x.tanh());
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
        // Извлекаем выход tanh, сохранённый в контексте (пока Mat<f32>)
        let y_mat = match ctx {
            DynamicContext::Mat(MatContext::Tanh { output }) => output.clone(),
            _ => panic!("Expected Tanh context"),
        };
        let go = grad_output.as_mat();
        let mut gi = grad_input.as_mat_mut();

        let dx = Mat::from_fn(y_mat.nrows(), y_mat.ncols(), |r, c| {
            let val = y_mat[(r, c)];
            go[(r, c)] * (1.0 - val * val)
        });
        gi.copy_from(&dx);
        Vec::new()
    }

    fn param_len(&self) -> usize { 0 }
    fn input_features(&self) -> usize { 0 }
    fn output_features(&self) -> usize { 0 }
}