// src/layers/memory/cpu/mod.rs

use crate::compute_manager::graph::types::DynamicContext;
use crate::compute_manager::matrix_buffer::MatrixBufferHandle;
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
    ) {
        // ВАЖНО: вычисляем размеры и берём копии скаляров ДО блокировки MemoryExecutor.
        let rows = input.rows();
        let features = self.features;
        let alpha = self.alpha;
        let ids = [input.id(), output.id()];

        let mut state_guard = self.state.lock().unwrap();
        let state = &mut *state_guard;

        input.memory().write().unwrap().with_cpu_slices_mut(&ids, |slices| {
            let (first, rest) = slices.split_at_mut(1);
            let x: &[f32] = &*first[0];
            let y: &mut [f32] = &mut *rest[0];

            // ============ Инициализация якорей ============
            // Выполняется один раз при первом прямом проходе, по образцу r = 0 батча.
            if !state.initialized {
                for c in 0..features {
                    let v = x[c * rows];
                    state.min_cells[c] = v;
                    state.max_cells[c] = v;
                }
                state.initialized = true;
            }

            // ============ Основной проход ============
            for r in 0..rows {
                for c in 0..features {
                    let idx = c * rows + r;
                    let x_val = x[idx];
                    let min_val = state.min_cells[c];
                    let max_val = state.max_cells[c];

                    let d_min = (x_val - min_val).abs();
                    let d_max = (x_val - max_val).abs();
                    let closest = if d_min <= d_max { min_val } else { max_val };

                    y[idx] = x_val + alpha * (closest - x_val);

                    // Обновление якорей в зависимости от положения x_val.
                    if x_val > max_val {
                        state.max_cells[c] = max_val + alpha * (x_val - max_val);
                    } else if x_val < min_val {
                        state.min_cells[c] = min_val + alpha * (x_val - min_val);
                    } else {
                        state.min_cells[c] = min_val + alpha * (x_val - min_val);
                        state.max_cells[c] = max_val + alpha * (x_val - max_val);
                    }
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
        let _input_handle = match bc {
            BufferedContext::Memory { input } => input,
            _ => panic!("Expected Memory context"),
        };

        let factor = 1.0 - self.alpha;
        let ids = [grad_output.id(), grad_input.id()];
        grad_output.memory().write().unwrap().with_cpu_slices_mut(&ids, |slices| {
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