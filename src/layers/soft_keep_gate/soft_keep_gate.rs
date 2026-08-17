// src/layers/soft_keep_gate/soft_keep_gate.rs

use crate::compute_manager::graph::types::DynamicContext;
use crate::compute_manager::matrix_buffer::MatrixBufferHandle;
use crate::layers::buffered_context::BufferedContext;
use crate::layers::{UniversalLayer, UniversalLayerBuffered};
use crate::model_plan::param_store::ParamSlice;

pub struct SoftKeepGate {
    pub in_features: usize,
    pub temperature: f32,
}

impl SoftKeepGate {
    pub fn new(in_features: usize, temperature: f32) -> Self {
        assert!(temperature > 0.0, "SoftKeepGate: temperature must be positive");
        Self { in_features, temperature }
    }
}

impl UniversalLayer for SoftKeepGate {
    fn as_soft_keep_gate(&self) -> Option<&SoftKeepGate> {
        Some(self)
    }

    fn param_len(&self) -> usize {
        self.in_features
    }

    fn input_features(&self) -> usize {
        self.in_features
    }

    fn output_features(&self) -> usize {
        self.in_features
    }
}

impl UniversalLayerBuffered for SoftKeepGate {
    fn forward_buffered(
        &self,
        input: &MatrixBufferHandle,
        output: &MatrixBufferHandle,
        params: &[f32],
        slice: &ParamSlice,
    ) {
        let input_guard = input.read();
        let src = input_guard.as_slice().expect("SoftKeepGate forward: expected CPU buffer");

        let mut output_guard = output.write();
        let dst = output_guard.as_slice_mut().expect("SoftKeepGate forward: expected CPU buffer");

        let rows = input.rows();
        let cols = input.cols();
        debug_assert_eq!(cols, self.in_features);

        let thresholds = &params[slice.start..slice.start + self.in_features];
        let tmp = self.temperature;

        debug_assert_eq!(src.len(), dst.len());

        for c in 0..cols {
            let threshold = thresholds[c];
            for r in 0..rows {
                let idx = c * rows + r;
                let x = src[idx];
                let abs_x = x.abs();
                let z = (threshold - abs_x) / tmp;
                let s = 1.0 / (1.0 + (-z).exp());
                dst[idx] = x * s;
            }
        }
    }

    fn backward_buffered(
        &self,
        ctx: &DynamicContext,
        grad_output: &MatrixBufferHandle,
        grad_input: &MatrixBufferHandle,
        params: &[f32],
        slice: &ParamSlice,
        grad_params: &MatrixBufferHandle,
    ) {
        let DynamicContext::Buffered(bc) = ctx;
        let input_handle = match bc {
            BufferedContext::SoftKeepGate { input } => input,
            _ => panic!("Expected SoftKeepGate context"),
        };

        let input_guard = input_handle.read();
        let x_slice = input_guard.as_slice().expect("SoftKeepGate backward: expected CPU buffer");

        let go_guard = grad_output.read();
        let go = go_guard.as_slice().expect("SoftKeepGate backward: expected CPU buffer");

        let mut gi_guard = grad_input.write();
        let gi = gi_guard.as_slice_mut().expect("SoftKeepGate backward: expected CPU buffer");

        let rows = grad_output.rows();
        let cols = grad_output.cols();
        debug_assert_eq!(cols, self.in_features);

        let thresholds = &params[slice.start..slice.start + self.in_features];
        let tmp = self.temperature;

        debug_assert_eq!(go.len(), gi.len());
        debug_assert_eq!(go.len(), x_slice.len());

        // Вычисляем градиент по входу и записываем пороговые градиенты напрямую в grad_params
        grad_params.with_cpu_data_mut(|grad_data| {
            for c in 0..cols {
                let threshold = thresholds[c];
                let mut d_thr = 0.0f32;
                for r in 0..rows {
                    let idx = c * rows + r;

                    let x_val = x_slice[idx];
                    let abs_x = x_val.abs();
                    let z = (threshold - abs_x) / tmp;
                    let s = 1.0 / (1.0 + (-z).exp());
                    let ds = s * (1.0 - s) / tmp;
                    let df_dx = s - abs_x * ds; // производная по x для SoftKeepGate

                    gi[idx] = go[idx] * df_dx;

                    // Градиент по порогам: d_s_dthr = ds
                    d_thr += go[idx] * x_val * ds;
                }
                grad_data[slice.start + c] = d_thr;
            }
        });
    }

    fn param_len(&self) -> usize {
        self.in_features
    }

    fn input_features(&self) -> usize {
        self.in_features
    }

    fn output_features(&self) -> usize {
        self.in_features
    }
}