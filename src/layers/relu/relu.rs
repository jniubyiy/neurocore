use crate::compute_manager::graph::types::DynamicContext;
use crate::compute_manager::matrix_buffer::MatrixBuffer;
use crate::layers::UniversalLayer;
use crate::layers::UniversalLayerBuffered;
use crate::model_plan::param_store::ParamSlice;
use crate::layers::mat_context::MatContext;
use faer::Mat;

pub struct ReLU;

impl ReLU {
    pub fn new() -> Self { Self }
}

// ---------------------------------------------------------------------------
// Старая реализация UniversalLayer (оставлена для GPU и обратной совместимости)
// ---------------------------------------------------------------------------

impl UniversalLayer for ReLU {
    fn forward_mat(
        &self,
        input: &Mat<f32>,
        _params: &[f32],
        _slice: &ParamSlice,
    ) -> (Mat<f32>, DynamicContext) {
        let output = input.map(|x| x.max(0.0));
        let ctx = DynamicContext::Mat(MatContext::ReLU { input: input.clone() });
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
            DynamicContext::Mat(MatContext::ReLU { input }) => input.clone(),
            _ => panic!("Expected ReLU context"),
        };
        let dx = Mat::from_fn(x_mat.nrows(), x_mat.ncols(), |r, c| {
            if x_mat[(r, c)] > 0.0 { delta[(r, c)] } else { 0.0 }
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
        let out_chunk = chunk.map(|x| x.max(0.0));
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
        DynamicContext::Mat(MatContext::ReLU { input: input_sample.clone() })
    }

    fn output_mat_shape(&self, _batch_size: usize) -> Mat<f32> {
        Mat::zeros(0, 0)
    }

    fn as_relu(&self) -> Option<&ReLU> {
        Some(self)
    }
}

// ---------------------------------------------------------------------------
// Новая реализация UniversalLayerBuffered (CPU‑путь с управляемыми буферами)
// ---------------------------------------------------------------------------

impl UniversalLayerBuffered for ReLU {
    fn forward_buffered(
        &self,
        input: &MatrixBuffer,
        output: &mut MatrixBuffer,
        _params: &[f32],
        _slice: &ParamSlice,
    ) {
        let inp = input.as_mat();
        let mut out = output.as_mat_mut();

        // Поэлементно применяем ReLU с записью в выходной буфер.
        // Можно скопировать результат из временного Mat, но чтобы избежать лишнего выделения,
        // используем faer-операцию map над MatRef? Однако map возвращает новый Mat.
        // Поэтому создадим временный Mat через Mat::from_fn, но затем скопируем в out.
        let temp = Mat::from_fn(inp.nrows(), inp.ncols(), |r, c| inp[(r, c)].max(0.0));
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
        // На данном этапе контекст всё ещё хранит Mat<f32>, извлекаем его
        let x_mat = match ctx {
            DynamicContext::Mat(MatContext::ReLU { input }) => input.clone(),
            _ => panic!("Expected ReLU context"),
        };
        let go = grad_output.as_mat();
        let mut gi = grad_input.as_mat_mut();

        let dx = Mat::from_fn(x_mat.nrows(), x_mat.ncols(), |r, c| {
            if x_mat[(r, c)] > 0.0 { go[(r, c)] } else { 0.0 }
        });
        gi.copy_from(&dx);
        // ReLU не имеет параметров, возвращаем пустой градиент
        Vec::new()
    }

    fn param_len(&self) -> usize { 0 }
    fn input_features(&self) -> usize { 0 }
    fn output_features(&self) -> usize { 0 }
}