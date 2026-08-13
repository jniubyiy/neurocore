// src/layers/dual_anchor/dual_anchor.rs

use crate::compute_manager::graph::types::DynamicContext;
use crate::compute_manager::matrix_buffer::MatrixBuffer;
use crate::layers::buffered_context::BufferedContext;
use crate::layers::UniversalLayer;
use crate::layers::UniversalLayerBuffered;
use crate::model_plan::param_store::ParamSlice;
use crate::layers::mat_context::MatContext;
use faer::Mat;

pub struct DualAnchor {
    pub features: usize,
}

impl DualAnchor {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        assert_eq!(in_features, out_features,
            "DualAnchor: in_features must equal out_features");
        Self { features: in_features }
    }

    fn get_params(&self, params: &[f32], slice: &ParamSlice) -> (Vec<f32>, Vec<f32>, f32) {
        let base = slice.start;
        let min_vals = params[base..base + self.features].to_vec();
        let max_vals = params[base + self.features..base + 2 * self.features].to_vec();
        let alpha = params[base + 2 * self.features];
        (min_vals, max_vals, alpha)
    }
}

// ---------------------------------------------------------------------------
// Старая реализация UniversalLayer (оставлена для GPU и обратной совместимости)
// ---------------------------------------------------------------------------

impl UniversalLayer for DualAnchor {
    fn forward_mat(
        &self,
        input: &Mat<f32>,
        params: &[f32],
        slice: &ParamSlice,
    ) -> (Mat<f32>, DynamicContext) {
        let (min_vals, max_vals, alpha) = self.get_params(params, slice);
        let batch = input.nrows();
        let features = self.features;

        let output = Mat::from_fn(batch, features, |r, c| {
            let x = input[(r, c)];
            let min_v = min_vals[c];
            let max_v = max_vals[c];
            let d_min = (x - min_v).abs();
            let d_max = (x - max_v).abs();
            let closest = if d_min <= d_max { min_v } else { max_v };
            x + alpha * (closest - x)
        });

        let ctx = DynamicContext::Mat(MatContext::DualAnchor1D { input: input.clone() });
        (output, ctx)
    }

    fn backward_mat(
        &self,
        ctx: &DynamicContext,
        delta: &Mat<f32>,
        params: &[f32],
        slice: &ParamSlice,
    ) -> (Mat<f32>, Vec<f32>) {
        let (min_vals, max_vals, alpha) = self.get_params(params, slice);
        let x_mat = match ctx {
            DynamicContext::Mat(MatContext::DualAnchor1D { input }) => input.clone(),
            _ => panic!("Expected DualAnchor1D context"),
        };

        let (dx, grad) = dual_anchor_backward_mat(&x_mat, delta, &min_vals, &max_vals, alpha);
        (dx, grad)
    }

    fn param_len(&self) -> usize {
        2 * self.features + 1
    }
    fn input_features(&self) -> usize { self.features }
    fn output_features(&self) -> usize { self.features }

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
        let (min_vals, max_vals, alpha) = self.get_params(params, slice);
        let chunk = input.submatrix(task_offset, 0, task_count, self.features);
        let out_chunk = Mat::from_fn(task_count, self.features, |r, c| {
            let x = chunk[(r, c)];
            let min_v = min_vals[c];
            let max_v = max_vals[c];
            let d_min = (x - min_v).abs();
            let d_max = (x - max_v).abs();
            let closest = if d_min <= d_max { min_v } else { max_v };
            x + alpha * (closest - x)
        });
        for r in 0..task_count {
            for c in 0..self.features {
                output[(task_offset + r, c)] = out_chunk[(r, c)];
            }
        }
    }

    fn create_sample_context(
        &self,
        input_sample: &Mat<f32>,
        _output_sample: &Mat<f32>,
    ) -> DynamicContext {
        DynamicContext::Mat(MatContext::DualAnchor1D { input: input_sample.clone() })
    }

    fn output_mat_shape(&self, batch_size: usize) -> Mat<f32> {
        Mat::zeros(batch_size, self.features)
    }

    fn as_dual_anchor(&self) -> Option<&DualAnchor> {
        Some(self)
    }
}

// ---------------------------------------------------------------------------
// Новая реализация UniversalLayerBuffered (CPU‑путь с управляемыми буферами)
// ---------------------------------------------------------------------------

impl UniversalLayerBuffered for DualAnchor {
    fn forward_buffered(
        &self,
        input: &MatrixBuffer,
        output: &mut MatrixBuffer,
        params: &[f32],
        slice: &ParamSlice,
    ) {
        let rows = input.rows();
        let cols = input.cols();
        debug_assert_eq!(cols, self.features);

        // Прямой доступ к параметрам без выделения Vec
        let base = slice.start;
        let min_vals = &params[base..base + self.features];
        let max_vals = &params[base + self.features..base + 2 * self.features];
        let alpha = params[base + 2 * self.features];

        let src = input.as_slice();
        let dst = output.as_slice_mut();
        debug_assert_eq!(src.len(), dst.len());

        // column-major: внешний цикл по признакам, внутренний по строкам
        for c in 0..cols {
            let min_v = min_vals[c];
            let max_v = max_vals[c];
            for r in 0..rows {
                let idx = c * rows + r;
                let x = src[idx];
                let d_min = (x - min_v).abs();
                let d_max = (x - max_v).abs();
                let closest = if d_min <= d_max { min_v } else { max_v };
                dst[idx] = x + alpha * (closest - x);
            }
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
        // Извлекаем буферизованный контекст
        let bc = match ctx {
            DynamicContext::Buffered(bc) => bc,
            _ => panic!("Expected Buffered context"),
        };
        let input_arc = match bc {
            BufferedContext::DualAnchor1D { input } => input,
            _ => panic!("Expected DualAnchor1D context"),
        };
        let input = input_arc.as_ref();

        let rows = grad_output.rows();
        let cols = grad_output.cols();
        debug_assert_eq!(cols, self.features);

        let base = slice.start;
        let min_vals = &params[base..base + self.features];
        let max_vals = &params[base + self.features..base + 2 * self.features];
        let alpha = params[base + 2 * self.features];

        let go = grad_output.as_slice();
        let gi = grad_input.as_slice_mut();
        let x_slice = input.as_slice();

        debug_assert_eq!(go.len(), gi.len());

        let mut d_min = vec![0.0f32; self.features];
        let mut d_max = vec![0.0f32; self.features];
        let mut d_alpha = 0.0f32;

        for c in 0..cols {
            let min_v = min_vals[c];
            let max_v = max_vals[c];
            for r in 0..rows {
                let idx = c * rows + r;

                let x_val = x_slice[idx];
                let d_min_abs = (x_val - min_v).abs();
                let d_max_abs = (x_val - max_v).abs();
                let is_min = d_min_abs <= d_max_abs;
                let gout = go[idx];

                // Градиент по входу
                gi[idx] = gout * (1.0 - alpha);

                // Градиенты по параметрам
                if is_min {
                    d_min[c] += gout * alpha;
                    d_alpha += gout * (min_v - x_val);
                } else {
                    d_max[c] += gout * alpha;
                    d_alpha += gout * (max_v - x_val);
                }
            }
        }

        let mut grad = Vec::with_capacity(2 * self.features + 1);
        grad.extend_from_slice(&d_min);
        grad.extend_from_slice(&d_max);
        grad.push(d_alpha);
        grad
    }

    fn param_len(&self) -> usize {
        2 * self.features + 1
    }
    fn input_features(&self) -> usize { self.features }
    fn output_features(&self) -> usize { self.features }
}

// Вспомогательная функция для старой матричной реализации
fn dual_anchor_backward_mat(
    x: &Mat<f32>,
    dout: &Mat<f32>,
    min_vals: &[f32],
    max_vals: &[f32],
    alpha: f32,
) -> (Mat<f32>, Vec<f32>) {
    let batch = x.nrows();
    let features = x.ncols();

    let mut dx = Mat::zeros(batch, features);
    let mut d_min = vec![0.0f32; features];
    let mut d_max = vec![0.0f32; features];
    let mut d_alpha = 0.0f32;

    for r in 0..batch {
        for c in 0..features {
            let x_val = x[(r, c)];
            let min_v = min_vals[c];
            let max_v = max_vals[c];
            let d_min_abs = (x_val - min_v).abs();
            let d_max_abs = (x_val - max_v).abs();
            let is_min = d_min_abs <= d_max_abs;
            let gout = dout[(r, c)];

            dx[(r, c)] += gout * (1.0 - alpha);

            if is_min {
                d_min[c] += gout * alpha;
                d_alpha += gout * (min_v - x_val);
            } else {
                d_max[c] += gout * alpha;
                d_alpha += gout * (max_v - x_val);
            }
        }
    }

    let mut grad = Vec::with_capacity(2 * features + 1);
    grad.extend_from_slice(&d_min);
    grad.extend_from_slice(&d_max);
    grad.push(d_alpha);
    (dx, grad)
}