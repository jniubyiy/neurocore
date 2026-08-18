// src/layers/memory/memory.rs

use std::sync::Mutex;

use crate::compute_manager::graph::types::DynamicContext;
use crate::compute_manager::matrix_buffer::MatrixBufferHandle;
use crate::layers::buffered_context::BufferedContext;
use crate::layers::{UniversalLayer, UniversalLayerBuffered};
use crate::model_plan::param_store::ParamSlice;

pub struct Memory {
    features: usize,
    pub alpha: f32,
    cells: Mutex<Vec<f32>>,
}

impl Memory {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        assert_eq!(in_features, out_features,
            "Memory: in_features must equal out_features");
        let mut cells = Vec::with_capacity(2 * in_features);
        cells.resize(in_features, f32::MAX);
        cells.resize(2 * in_features, f32::MIN);
        Self {
            features: in_features,
            alpha: 0.1,
            cells: Mutex::new(cells),
        }
    }
}

impl UniversalLayer for Memory {
    fn as_memory(&self) -> Option<&Memory> {
        Some(self)
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

impl UniversalLayerBuffered for Memory {
    fn forward_buffered(
        &self,
        input: &MatrixBufferHandle,
        output: &MatrixBufferHandle,
        _params: &MatrixBufferHandle,
        _slice: &ParamSlice,
    ) {
        let ids = [input.id(), output.id()];
        let mut cells_guard = self.cells.lock().unwrap();
        let cells = &mut *cells_guard;

        input.memory().lock().unwrap().with_cpu_slices_mut(&ids, |slices| {
            let (first, rest) = slices.split_at_mut(1);
            let x: &[f32] = &*first[0];
            let y: &mut [f32] = &mut *rest[0];
            let rows = input.rows();
            let features = self.features;

            for r in 0..rows {
                for c in 0..features {
                    let idx = c * rows + r;
                    let x_val = x[idx];
                    let min_idx = c;
                    let max_idx = features + c;
                    let min_val = cells[min_idx];
                    let max_val = cells[max_idx];

                    let d_min = (x_val - min_val).abs();
                    let d_max = (x_val - max_val).abs();
                    let closest = if d_min <= d_max { min_val } else { max_val };
                    y[idx] = x_val + self.alpha * (closest - x_val);

                    if x_val > max_val {
                        cells[max_idx] += self.alpha * (x_val - max_val);
                    } else if x_val < min_val {
                        cells[min_idx] += self.alpha * (x_val - min_val);
                    } else {
                        cells[min_idx] += self.alpha * (x_val - min_val);
                        cells[max_idx] += self.alpha * (x_val - max_val);
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
        grad_output.memory().lock().unwrap().with_cpu_slices_mut(&ids, |slices| {
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