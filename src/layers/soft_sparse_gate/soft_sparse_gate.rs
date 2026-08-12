use crate::compute_manager::graph::types::DynamicContext;
use crate::compute_manager::matrix_buffer::MatrixBuffer;
use crate::layers::UniversalLayer;
use crate::layers::UniversalLayerBuffered;
use crate::model_plan::param_store::ParamSlice;
use crate::layers::mat_context::MatContext;
use faer::Mat;

pub struct SoftSparseGate {
    pub in_features: usize,
    pub temperature: f32,
}

impl SoftSparseGate {
    pub fn new(in_features: usize, temperature: f32) -> Self {
        assert!(temperature > 0.0, "SoftSparseGate: temperature must be positive");
        Self { in_features, temperature }
    }
}

// ---------------------------------------------------------------------------
// Старая реализация UniversalLayer (оставлена для GPU и обратной совместимости)
// ---------------------------------------------------------------------------

impl UniversalLayer for SoftSparseGate {
    fn forward_mat(
        &self,
        input: &Mat<f32>,
        params: &[f32],
        slice: &ParamSlice,
    ) -> (Mat<f32>, DynamicContext) {
        let thresholds = &params[slice.start..slice.start + self.in_features];
        let tmp = self.temperature;

        let output = Mat::from_fn(input.nrows(), input.ncols(), |r, c| {
            let x = input[(r, c)];
            let z = (x.abs() - thresholds[c]) / tmp;
            x / (1.0 + (-z).exp())
        });

        let ctx = DynamicContext::Mat(MatContext::SoftSparseGate { input: input.clone() });
        (output, ctx)
    }

    fn backward_mat(
        &self,
        ctx: &DynamicContext,
        delta: &Mat<f32>,
        params: &[f32],
        slice: &ParamSlice,
    ) -> (Mat<f32>, Vec<f32>) {
        let thresholds = &params[slice.start..slice.start + self.in_features];
        let tmp = self.temperature;

        let x_mat = match ctx {
            DynamicContext::Mat(MatContext::SoftSparseGate { input }) => input.clone(),
            _ => panic!("Expected SoftSparseGate context"),
        };

        let (dx, d_thr) = soft_sparse_gate_backward_mat(&x_mat, delta, thresholds, tmp);
        (dx, d_thr)
    }

    fn param_len(&self) -> usize { self.in_features }
    fn input_features(&self) -> usize { self.in_features }
    fn output_features(&self) -> usize { self.in_features }

    fn total_tasks(&self, batch_size: usize) -> usize { batch_size }

    fn execute_tasks(
        &self,
        input: &Mat<f32>,
        output: &mut Mat<f32>,
        task_offset: usize,
        task_count: usize,
        params: &[f32],
        slice: &ParamSlice,
    ) {
        let thresholds = &params[slice.start..slice.start + self.in_features];
        let tmp = self.temperature;
        let chunk = input.submatrix(task_offset, 0, task_count, self.in_features);
        let out_chunk = Mat::from_fn(task_count, self.in_features, |r, c| {
            let x = chunk[(r, c)];
            let z = (x.abs() - thresholds[c]) / tmp;
            x / (1.0 + (-z).exp())
        });
        for r in 0..task_count {
            for c in 0..self.in_features {
                output[(task_offset + r, c)] = out_chunk[(r, c)];
            }
        }
    }

    fn create_sample_context(
        &self,
        input_sample: &Mat<f32>,
        _output_sample: &Mat<f32>,
    ) -> DynamicContext {
        DynamicContext::Mat(MatContext::SoftSparseGate { input: input_sample.clone() })
    }

    fn output_mat_shape(&self, batch_size: usize) -> Mat<f32> {
        Mat::zeros(batch_size, self.in_features)
    }

    fn as_soft_sparse_gate(&self) -> Option<&SoftSparseGate> {
        Some(self)
    }
}

// ---------------------------------------------------------------------------
// Новая реализация UniversalLayerBuffered (CPU‑путь с управляемыми буферами)
// ---------------------------------------------------------------------------

impl UniversalLayerBuffered for SoftSparseGate {
    fn forward_buffered(
        &self,
        input: &MatrixBuffer,
        output: &mut MatrixBuffer,
        params: &[f32],
        slice: &ParamSlice,
    ) {
        let rows = input.rows();
        let cols = input.cols();
        debug_assert_eq!(cols, self.in_features);

        let thresholds = &params[slice.start..slice.start + self.in_features];
        let tmp = self.temperature;

        let src = input.as_slice();
        let dst = output.as_slice_mut();
        debug_assert_eq!(src.len(), dst.len());

        for idx in 0..src.len() {
            let r = idx % rows;
            let c = idx / rows;

            let x = src[idx];
            let abs_x = x.abs();
            let z = (abs_x - thresholds[c]) / tmp;
            let s = 1.0 / (1.0 + (-z).exp());
            dst[idx] = x * s;
        }
    }

    fn backward_buffered(
        &self,
        ctx: &DynamicContext,
        grad_output: &MatrixBuffer,
        grad_input: &mut MatrixBuffer,
        params: &[f32],
        slice: &ParamSlice,
    ) -> Vec<f32> {
        // Извлекаем входную матрицу из контекста без копирования
        let x_mat = match ctx {
            DynamicContext::Mat(MatContext::SoftSparseGate { input }) => input,
            _ => panic!("Expected SoftSparseGate context"),
        };

        let rows = grad_output.rows();
        let cols = grad_output.cols();
        debug_assert_eq!(cols, self.in_features);

        let thresholds = &params[slice.start..slice.start + self.in_features];
        let tmp = self.temperature;

        let go = grad_output.as_slice();
        let gi = grad_input.as_slice_mut();
        debug_assert_eq!(go.len(), gi.len());

        let mut d_thr = vec![0.0f32; self.in_features];

        for idx in 0..go.len() {
            let r = idx % rows;
            let c = idx / rows;

            let x_val = x_mat[(r, c)];
            let abs_x = x_val.abs();
            let z = (abs_x - thresholds[c]) / tmp;
            let s = 1.0 / (1.0 + (-z).exp());
            let ds = s * (1.0 - s) / tmp;
            let df_dx = s + abs_x * ds;

            gi[idx] = go[idx] * df_dx;

            // Градиент по порогам: d_s_dthr = -ds
            d_thr[c] += -go[idx] * x_val * ds;
        }

        d_thr
    }

    fn param_len(&self) -> usize { self.in_features }
    fn input_features(&self) -> usize { self.in_features }
    fn output_features(&self) -> usize { self.in_features }
}

fn soft_sparse_gate_backward_mat(
    x: &Mat<f32>,
    dout: &Mat<f32>,
    thresholds: &[f32],
    temperature: f32,
) -> (Mat<f32>, Vec<f32>) {
    let rows = x.nrows();
    let cols = x.ncols();
    assert_eq!(cols, thresholds.len());

    let mut dx = Mat::zeros(rows, cols);
    let mut d_thr = vec![0.0f32; cols];

    for r in 0..rows {
        for c in 0..cols {
            let x_val = x[(r, c)];
            let abs_x = x_val.abs();
            let z = (abs_x - thresholds[c]) / temperature;
            let s = 1.0 / (1.0 + (-z).exp());
            let ds = s * (1.0 - s) / temperature;
            let df_dx = s + abs_x * ds;
            dx[(r, c)] = dout[(r, c)] * df_dx;

            let d_s_dthr = -ds;
            d_thr[c] += dout[(r, c)] * x_val * d_s_dthr;
        }
    }

    (dx, d_thr)
}