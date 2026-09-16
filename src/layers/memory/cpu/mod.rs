// src/layers/memory/cpu/mod.rs

use crate::compute_manager::graph::types::DynamicContext;
use crate::compute_manager::matrix_buffer::{MatrixBufferHandle, TempMatrixPool};
use crate::layers::buffered_context::BufferedContext;
use crate::layers::UniversalLayerBuffered;
use crate::model_plan::param_store::ParamSlice;

use super::super::memory::Memory;

impl UniversalLayerBuffered for Memory {
    fn forward_buffered(
        &self,
        input: &MatrixBufferHandle,
        output: &MatrixBufferHandle,
        _params: &MatrixBufferHandle,
        _slice: &ParamSlice,
        pool: &mut TempMatrixPool,
    ) -> BufferedContext {
        let rows = input.rows();
        let features = self.features;
        let alpha = self.alpha;

        // ------------------------------------------------------------------
        // Шаг 1. Создаём state-буферы якорей из пула.
        //
        //   min_cells: (features × 1) — минимальные якоря по признакам.
        //   max_cells: (features × 1) — максимальные якоря по признакам.
        //
        // Буферы per-chunk: создаются на каждый forward, живут в контексте
        // до конца backward. Пул переиспользует их между вызовами.
        // ------------------------------------------------------------------
        let min_cells = pool.acquire(features, 1);
        let max_cells = pool.acquire(features, 1);

        // Инициализируем якоря по строке r = 0 входного чанка. Это те же
        // значения, которые раньше брались из первой строки в первом вызове
        // forward; теперь они берутся из первой строки каждого чанка.
        {
            let src = input.read();
            let src_slice = src
                .as_slice()
                .expect("Memory forward: input must be CPU");

            let mut min_guard = min_cells.write();
            let min_slice = min_guard
                .as_slice_mut()
                .expect("Memory forward: min_cells must be CPU");

            let mut max_guard = max_cells.write();
            let max_slice = max_guard
                .as_slice_mut()
                .expect("Memory forward: max_cells must be CPU");

            for c in 0..features {
                let v = src_slice[c * rows];
                min_slice[c] = v;
                max_slice[c] = v;
            }
        }

        // ------------------------------------------------------------------
        // Шаг 2. Основной проход: вычисляем y и обновляем якоря in-place.
        //
        // Порядок обхода — r = 0..rows, как в исходной реализации. Это
        // важно: обновление якорей некоммутативно, и результат для строки r
        // зависит от того, какие строки прошли до неё.
        // ------------------------------------------------------------------
        let ids = [
            input.id(),
            output.id(),
            min_cells.id(),
            max_cells.id(),
        ];
        input
            .memory()
            .write()
            .unwrap()
            .with_cpu_slices_mut(&ids, |slices| {
                let (first, rest) = slices.split_at_mut(1);
                let x: &[f32] = &*first[0];

                let (second, rest) = rest.split_at_mut(1);
                let y: &mut [f32] = &mut *second[0];

                let (third, rest) = rest.split_at_mut(1);
                let min_slice: &mut [f32] = &mut *third[0];

                let (fourth, _) = rest.split_at_mut(1);
                let max_slice: &mut [f32] = &mut *fourth[0];

                for r in 0..rows {
                    for c in 0..features {
                        let idx = c * rows + r;
                        let x_val = x[idx];
                        let min_val = min_slice[c];
                        let max_val = max_slice[c];

                        let d_min = (x_val - min_val).abs();
                        let d_max = (x_val - max_val).abs();
                        let closest = if d_min <= d_max { min_val } else { max_val };

                        y[idx] = x_val + alpha * (closest - x_val);

                        // Обновление якорей в зависимости от положения x_val.
                        if x_val > max_val {
                            max_slice[c] = max_val + alpha * (x_val - max_val);
                        } else if x_val < min_val {
                            min_slice[c] = min_val + alpha * (x_val - min_val);
                        } else {
                            min_slice[c] = min_val + alpha * (x_val - min_val);
                            max_slice[c] = max_val + alpha * (x_val - max_val);
                        }
                    }
                }
            });

        // ------------------------------------------------------------------
        // Шаг 3. Возвращаем контекст с дескрипторами входного буфера и
        // обоих state-буферов. Backward state не использует, но контекст
        // обязан содержать эти дескрипторы: пока контекст жив, буферы
        // не возвращаются в пул.
        // ------------------------------------------------------------------
        BufferedContext::Memory {
            input: input.clone(),
            min_cells,
            max_cells,
        }
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
        // Backward у Memory — линейная передача градиента с коэффициентом
        // (1 − alpha). State не участвует, но контекст читаем, чтобы
        // сохранить контракт: любой Buffered backward начинается с
        // разбора DynamicContext::Buffered.
        let DynamicContext::Buffered(bc) = ctx;
        let _input_handle = match bc {
            BufferedContext::Memory { input, .. } => input,
            _ => panic!("Expected Memory Buffered context"),
        };

        let factor = 1.0 - self.alpha;
        let ids = [grad_output.id(), grad_input.id()];
        grad_output
            .memory()
            .write()
            .unwrap()
            .with_cpu_slices_mut(&ids, |slices| {
                let (first, rest) = slices.split_at_mut(1);
                let go: &[f32] = &*first[0];
                let gi: &mut [f32] = &mut *rest[0];
                for i in 0..go.len() {
                    gi[i] = go[i] * factor;
                }
            });
    }

    fn param_len(&self) -> usize {
        0
    }

    fn input_features(&self) -> usize {
        self.features
    }

    fn output_features(&self) -> usize {
        self.features
    }
}