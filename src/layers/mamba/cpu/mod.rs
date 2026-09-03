// src/layers/mamba/cpu/mod.rs

use crate::compute_manager::graph::types::DynamicContext;
use crate::compute_manager::matrix_buffer::MatrixBufferHandle;
use crate::layers::buffered_context::BufferedContext;
use crate::layers::UniversalLayerBuffered;
use crate::model_plan::param_store::ParamSlice;

use super::super::mamba::Mamba;

impl UniversalLayerBuffered for Mamba {
    fn forward_buffered(
        &self,
        input: &MatrixBufferHandle,
        output: &MatrixBufferHandle,
        params: &MatrixBufferHandle,
        slice: &ParamSlice,
    ) {
        let batch = input.rows();
        let features = input.cols();
        let d = self.input_dim;
        let seq = self.seq_len;
        let n = self.state_dim;
        debug_assert_eq!(features, seq * d);
        debug_assert!(slice.start + self.param_len() <= params.rows() * params.cols());

        let ids = [input.id(), output.id(), params.id()];
        input.memory().write().unwrap().with_cpu_slices_mut(&ids, |slices| {
            let (first, rest) = slices.split_at_mut(1);
            let x: &[f32] = &*first[0];
            let (second, rest) = rest.split_at_mut(1);
            let y: &mut [f32] = &mut *second[0];
            let p: &[f32] = &*rest[0];

            let base = slice.start;
            let a_start = base;
            let b_start = a_start + n * n;
            let c_start = b_start + n;
            let d_param = p[c_start + n];       // D
            let delta = p[c_start + n + 1];     // Δ

            // Для простоты: один набор B, C на все признаки
            // A_bar = exp(delta * A) — поэлементно (упрощённо)
            // B_bar = delta * B

            let mut h = vec![0.0f32; batch * n];
            for t in 0..seq {
                for r in 0..batch {
                    // x_t для этого батча (признак d? если d > 1, берём первый признак?)
                    // В текущей упрощённой версии предполагаем d = 1, чтобы не усложнять.
                    // Для d > 1 нужно отдельные B, C для каждого признака.
                    if d != 1 {
                        panic!("Mamba CPU forward currently supports input_dim == 1");
                    }
                    let x_t = x[(t * 1) * batch + r]; // так как d=1

                    // Вычисляем B_bar = delta * B
                    // A_bar = exp(delta * A) поэлементно
                    // h_next = A_bar * h + B_bar * x_t
                    let mut h_new = vec![0.0f32; n];
                    for i in 0..n {
                        let mut sum = 0.0;
                        for j in 0..n {
                            let a_ij = p[a_start + i * n + j];
                            let a_bar_ij = (delta * a_ij).exp();
                            sum += a_bar_ij * h[r * n + j];
                        }
                        let b_i = p[b_start + i];
                        h_new[i] = sum + delta * b_i * x_t;
                    }
                    // обновляем h
                    for i in 0..n { h[r * n + i] = h_new[i]; }

                    // Выход y_t = C^T h + D x_t
                    let mut y_t = d_param * x_t;
                    for i in 0..n {
                        y_t += p[c_start + i] * h[r * n + i];
                    }
                    y[(t * 1) * batch + r] = y_t;
                }
            }
        });
    }

    fn backward_buffered(
        &self,
        _ctx: &DynamicContext,
        _grad_output: &MatrixBufferHandle,
        _grad_input: &MatrixBufferHandle,
        _params: &MatrixBufferHandle,
        _slice: &ParamSlice,
        _grad_params: &MatrixBufferHandle,
    ) {
        panic!("Mamba backward is not implemented yet");
    }

    fn param_len(&self) -> usize {
        let n = self.state_dim;
        n * n + 2 * n + 2
    }

    fn input_features(&self) -> usize {
        self.seq_len * self.input_dim
    }

    fn output_features(&self) -> usize {
        self.seq_len * self.input_dim
    }
}