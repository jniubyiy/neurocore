// src/layers/ind_rnn/cpu/mod.rs

use crate::compute_manager::graph::types::DynamicContext;
use crate::compute_manager::matrix_buffer::MatrixBufferHandle;
use crate::layers::buffered_context::BufferedContext;
use crate::layers::UniversalLayerBuffered;
use crate::model_plan::param_store::ParamSlice;

use super::super::ind_rnn::ind_rnn::{IndRNN, IndRNNForwardCache};

impl UniversalLayerBuffered for IndRNN {
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
        debug_assert_eq!(features, seq * d);
        debug_assert_eq!(output.rows(), batch);
        debug_assert_eq!(output.cols(), features);
        debug_assert!(slice.start + self.param_len() <= params.rows() * params.cols());

        let ids = [input.id(), output.id(), params.id()];
        let (input_copy, hidden_states) = input
            .memory()
            .write()
            .unwrap()
            .with_cpu_slices_mut(&ids, |slices| {
                let (first, rest) = slices.split_at_mut(1);
                let x: &[f32] = &*first[0];
                let (second, rest) = rest.split_at_mut(1);
                let y: &mut [f32] = &mut *second[0];
                let p: &[f32] = &*rest[0];

                let base = slice.start;
                let w_start = base;
                let u_start = w_start + d * d;
                let b_start = u_start + d;

                let mut hidden_states = vec![0.0f32; batch * seq * d];
                let input_copy = x.to_vec();

                let mut h_prev = vec![0.0f32; batch * d];

                for t in 0..seq {
                    for r in 0..batch {
                        for j in 0..d {
                            let mut sum = p[b_start + j];
                            for i in 0..d {
                                let x_idx = (t * d + i) * batch + r;
                                sum += x[x_idx] * p[w_start + j * d + i];
                            }
                            sum += p[u_start + j] * h_prev[r * d + j];
                            let h = if sum > 0.0 { sum } else { 0.0 };
                            hidden_states[(r * seq + t) * d + j] = h;
                            h_prev[r * d + j] = h;
                            let out_idx = (t * d + j) * batch + r;
                            y[out_idx] = h;
                        }
                    }
                }

                (input_copy, hidden_states)
            });

        self.store_cache(IndRNNForwardCache {
            input: input_copy,
            hidden_states,
        });
    }

    fn backward_buffered(
        &self,
        ctx: &DynamicContext,
        grad_output: &MatrixBufferHandle,
        grad_input: &MatrixBufferHandle,
        params: &MatrixBufferHandle,
        slice: &ParamSlice,
        grad_params: &MatrixBufferHandle,
    ) {
        let DynamicContext::Buffered(bc) = ctx;
        let _input_handle = match bc {
            BufferedContext::IndRNN { input, .. } => input,
            _ => panic!("Expected IndRNN context"),
        };

        let batch = grad_output.rows();
        let d = self.input_dim;
        let seq = self.seq_len;

        debug_assert_eq!(grad_output.cols(), seq * d);
        debug_assert_eq!(grad_input.rows(), batch);
        debug_assert_eq!(grad_input.cols(), seq * d);
        debug_assert!(slice.start + self.param_len() <= params.rows() * params.cols());
        debug_assert!(slice.start + self.param_len() <= grad_params.rows() * grad_params.cols());

        // Извлекаем кэш и инвалидируем состояние.
        // std::mem::replace позволяет избежать клонирования больших векторов.
        let cache = {
            let mut guard = self.state.write().unwrap();
            assert!(
                guard.valid,
                "IndRNN backward called without forward cache"
            );
            guard.valid = false;
            std::mem::replace(
                &mut guard.cache,
                IndRNNForwardCache {
                    input: Vec::new(),
                    hidden_states: Vec::new(),
                },
            )
        };

        let ids = [
            grad_output.id(),
            grad_input.id(),
            params.id(),
            grad_params.id(),
        ];

        grad_output
            .memory()
            .write()
            .unwrap()
            .with_cpu_slices_mut(&ids, |slices| {
                let (first, rest) = slices.split_at_mut(1);
                let go: &[f32] = &*first[0];
                let (second, rest) = rest.split_at_mut(1);
                let gi: &mut [f32] = &mut *second[0];
                let (third, rest) = rest.split_at_mut(1);
                let p: &[f32] = &*third[0];
                let gp: &mut [f32] = &mut *rest[0];

                let base = slice.start;
                let w_start = base;
                let u_start = w_start + d * d;
                let b_start = u_start + d;

                for i in 0..self.param_len() {
                    gp[base + i] = 0.0;
                }

                let mut grad_w = vec![0.0f32; d * d];
                let mut grad_u = vec![0.0f32; d];
                let mut grad_b = vec![0.0f32; d];

                for i in 0..(batch * seq * d) {
                    gi[i] = 0.0;
                }

                let mut delta_next = vec![0.0f32; batch * d];

                for t in (0..seq).rev() {
                    for r in 0..batch {
                        for j in 0..d {
                            let h_t = cache.hidden_states[(r * seq + t) * d + j];
                            let grad_out_t = go[(t * d + j) * batch + r];

                            let d_l_dh =
                                grad_out_t + p[u_start + j] * delta_next[r * d + j];

                            let d_relu = if h_t > 0.0 { 1.0 } else { 0.0 };

                            let delta_t = d_l_dh * d_relu;

                            grad_b[j] += delta_t;

                            if t > 0 {
                                let h_prev =
                                    cache.hidden_states[(r * seq + t - 1) * d + j];
                                grad_u[j] += delta_t * h_prev;
                            }

                            for i in 0..d {
                                let x_t_i = cache.input[(t * d + i) * batch + r];
                                grad_w[j * d + i] += delta_t * x_t_i;
                                gi[(t * d + i) * batch + r] +=
                                    delta_t * p[w_start + j * d + i];
                            }

                            delta_next[r * d + j] = delta_t;
                        }
                    }
                }

                for j in 0..d {
                    gp[b_start + j] = grad_b[j];
                    gp[u_start + j] = grad_u[j];
                }
                for i in 0..(d * d) {
                    gp[w_start + i] = grad_w[i];
                }
            });
    }

    fn param_len(&self) -> usize {
        let d = self.input_dim;
        d * d + 2 * d
    }

    fn input_features(&self) -> usize {
        self.seq_len * self.input_dim
    }

    fn output_features(&self) -> usize {
        self.seq_len * self.input_dim
    }
}