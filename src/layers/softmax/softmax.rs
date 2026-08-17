// src/layers/softmax/softmax.rs

use crate::compute_manager::graph::types::DynamicContext;
use crate::compute_manager::matrix_buffer::MatrixBufferHandle;
use crate::layers::buffered_context::BufferedContext;
use crate::layers::{UniversalLayer, UniversalLayerBuffered};
use crate::model_plan::param_store::ParamSlice;

pub struct Softmax;

impl Softmax {
    pub fn new() -> Self {
        Self
    }
}

impl UniversalLayer for Softmax {
    fn as_softmax(&self) -> Option<&Softmax> {
        Some(self)
    }
}

impl UniversalLayerBuffered for Softmax {
    fn forward_buffered(
        &self,
        input: &MatrixBufferHandle,
        output: &MatrixBufferHandle,
        _params: &[f32],
        _slice: &ParamSlice,
    ) {
        let src_guard = input.read();
        let src = src_guard.as_slice().expect("Softmax forward: expected CPU buffer");

        let mut dst_guard = output.write();
        let dst = dst_guard.as_slice_mut().expect("Softmax forward: expected CPU buffer");

        let rows = input.rows();
        let cols = input.cols();

        debug_assert_eq!(src.len(), rows * cols);
        debug_assert_eq!(dst.len(), rows * cols);

        // Для каждой строки (batch) вычисляем stable softmax
        for r in 0..rows {
            // 1. Находим максимум
            let mut max_val = f32::NEG_INFINITY;
            for c in 0..cols {
                let idx = c * rows + r;
                if src[idx] > max_val {
                    max_val = src[idx];
                }
            }

            // 2. Считаем сумму экспонент
            let mut sum_exp = 0.0f32;
            for c in 0..cols {
                let idx = c * rows + r;
                sum_exp += (src[idx] - max_val).exp();
            }

            // 3. Записываем нормализованные значения
            for c in 0..cols {
                let idx = c * rows + r;
                dst[idx] = (src[idx] - max_val).exp() / sum_exp;
            }
        }
    }

    fn backward_buffered(
        &self,
        ctx: &DynamicContext,
        grad_output: &MatrixBufferHandle,
        grad_input: &MatrixBufferHandle,
        _params: &[f32],
        _slice: &ParamSlice,
        _grad_params: &MatrixBufferHandle,
    ) {
        let DynamicContext::Buffered(bc) = ctx;
        let output_handle = match bc {
            BufferedContext::Softmax { output } => output,
            _ => panic!("Expected Softmax context"),
        };

        let output_guard = output_handle.read();
        let y_slice = output_guard.as_slice().expect("Softmax backward: expected CPU buffer");

        let go_guard = grad_output.read();
        let go = go_guard.as_slice().expect("Softmax backward: expected CPU buffer");

        let mut gi_guard = grad_input.write();
        let gi = gi_guard.as_slice_mut().expect("Softmax backward: expected CPU buffer");

        let rows = grad_output.rows();
        let cols = grad_output.cols();

        debug_assert_eq!(go.len(), rows * cols);
        debug_assert_eq!(gi.len(), rows * cols);
        debug_assert_eq!(y_slice.len(), rows * cols);

        // Для каждой строки
        for r in 0..rows {
            // Вычисляем dot = sum(y * grad_output) по строке
            let mut dot = 0.0f32;
            for c in 0..cols {
                let idx = c * rows + r;
                dot += y_slice[idx] * go[idx];
            }

            // Вычисляем градиент
            for c in 0..cols {
                let idx = c * rows + r;
                let y_val = y_slice[idx];
                gi[idx] = y_val * (go[idx] - dot);
            }
        }
    }

    fn param_len(&self) -> usize {
        0
    }

    fn input_features(&self) -> usize {
        0
    }

    fn output_features(&self) -> usize {
        0
    }
}