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
        _params: &MatrixBufferHandle,
        _slice: &ParamSlice,
    ) {
        let ids = [input.id(), output.id()];
        input.memory().lock().unwrap().with_cpu_slices_mut(&ids, |slices| {
            let (first, rest) = slices.split_at_mut(1);
            let x: &[f32] = &*first[0];
            let y: &mut [f32] = &mut *rest[0];
            let rows = input.rows();
            let cols = input.cols();

            for r in 0..rows {
                // 1. Находим максимум
                let mut max_val = f32::NEG_INFINITY;
                for c in 0..cols {
                    let idx = c * rows + r;
                    if x[idx] > max_val {
                        max_val = x[idx];
                    }
                }

                // 2. Считаем сумму экспонент
                let mut sum_exp = 0.0f32;
                for c in 0..cols {
                    let idx = c * rows + r;
                    sum_exp += (x[idx] - max_val).exp();
                }

                // 3. Записываем нормализованные значения
                for c in 0..cols {
                    let idx = c * rows + r;
                    y[idx] = (x[idx] - max_val).exp() / sum_exp;
                }
            }
        });
    }

    fn backward_buffered(
        &self,
        ctx: &DynamicContext,
        grad_output: &MatrixBufferHandle,
        grad_input: &MatrixBufferHandle,
        _params: &MatrixBufferHandle,
        _slice: &ParamSlice,
        _grad_params: &MatrixBufferHandle,
    ) {
        let DynamicContext::Buffered(bc) = ctx;
        let output_handle = match bc {
            BufferedContext::Softmax { output } => output,
            _ => panic!("Expected Softmax context"),
        };

        let ids = [output_handle.id(), grad_output.id(), grad_input.id()];
        output_handle.memory().lock().unwrap().with_cpu_slices_mut(&ids, |slices| {
            let (first, rest) = slices.split_at_mut(1);
            let y: &[f32] = &*first[0];
            let (second, rest) = rest.split_at_mut(1);
            let go: &[f32] = &*second[0];
            let gi: &mut [f32] = &mut *rest[0];
            let rows = grad_output.rows();
            let cols = grad_output.cols();

            for r in 0..rows {
                // Вычисляем dot = sum(y * grad_output) по строке
                let mut dot = 0.0f32;
                for c in 0..cols {
                    let idx = c * rows + r;
                    dot += y[idx] * go[idx];
                }

                // Вычисляем градиент
                for c in 0..cols {
                    let idx = c * rows + r;
                    let y_val = y[idx];
                    gi[idx] = y_val * (go[idx] - dot);
                }
            }
        });
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