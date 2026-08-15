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
        _params: &[f32],
        _slice: &ParamSlice,
    ) {
        let input_guard = input.read();
        let src = input_guard.as_slice().expect("Memory forward: expected CPU buffer");

        let mut output_guard = output.write();
        let dst = output_guard.as_slice_mut().expect("Memory forward: expected CPU buffer");

        let features = self.features;
        let mut cells = self.cells.lock().unwrap();

        debug_assert_eq!(src.len(), dst.len());

        for idx in 0..src.len() {
            let c = idx / input.rows(); // column index (feature)

            let x = src[idx];
            let min_idx = c;
            let max_idx = features + c;
            let min_val = cells[min_idx];
            let max_val = cells[max_idx];

            let d_min = (x - min_val).abs();
            let d_max = (x - max_val).abs();
            let closest = if d_min <= d_max { min_val } else { max_val };
            dst[idx] = x + self.alpha * (closest - x);

            if x > max_val {
                cells[max_idx] += self.alpha * (x - max_val);
            } else if x < min_val {
                cells[min_idx] += self.alpha * (x - min_val);
            } else {
                cells[min_idx] += self.alpha * (x - min_val);
                cells[max_idx] += self.alpha * (x - max_val);
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
    ) -> Vec<f32> {
        let bc = match ctx {
            DynamicContext::Buffered(bc) => bc,
        };
        let _input_handle = match bc {
            BufferedContext::Memory { input } => input,
            _ => panic!("Expected Memory context"),
        };

        let factor = 1.0 - self.alpha;
        let go_guard = grad_output.read();
        let go = go_guard.as_slice().expect("Memory backward: expected CPU buffer");

        let mut gi_guard = grad_input.write();
        let gi = gi_guard.as_slice_mut().expect("Memory backward: expected CPU buffer");

        debug_assert_eq!(go.len(), gi.len());

        for (out, &in_val) in gi.iter_mut().zip(go.iter()) {
            *out = in_val * factor;
        }

        Vec::new()
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