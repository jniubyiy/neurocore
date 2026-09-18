// src/layers/ind_rnn/cpu/mod.rs

use crate::compute_manager::core::dynamic_context::DynamicContext;
use crate::compute_manager::operators_v2::memory_v2::buffer::{MatrixBufferHandle, TempMatrixPool};
use crate::layers::buffered_context::BufferedContext;
use crate::layers::UniversalLayerBuffered;
use crate::model_plan::param_store::ParamSlice;

use super::super::ind_rnn::IndRNN;

impl UniversalLayerBuffered for IndRNN {
    fn forward_buffered(
        &self,
        input: &MatrixBufferHandle,
        output: &MatrixBufferHandle,
        params: &MatrixBufferHandle,
        slice: &ParamSlice,
        pool: &mut TempMatrixPool,
    ) -> BufferedContext {
        let batch = input.rows();
        let d = self.input_dim;
        let seq = self.seq_len;
        let features = seq * d;

        debug_assert_eq!(input.cols(), features);
        debug_assert_eq!(output.rows(), batch);
        debug_assert_eq!(output.cols(), features);
        debug_assert!(
            slice.start + self.param_len() <= params.rows() * params.cols(),
            "IndRNN: parameter slice out of bounds"
        );

        // ------------------------------------------------------------------
        // Per-chunk state-буфер:
        //   h_all: (batch * seq_len × input_dim) — все скрытые состояния.
        //
        // Column-major раскладка:
        //   элемент (r, t, j) лежит по адресу j * (batch * seq) + (r * seq + t).
        //
        // Буфер создаётся из пула на каждый forward, живёт в контексте до
        // конца backward, затем освобождается.
        // ------------------------------------------------------------------
        let h_all = pool.acquire(batch * seq, d);

        let ids = [
            input.id(),
            output.id(),
            params.id(),
            h_all.id(),
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
                let p: &[f32] = &*third[0];

                let (fourth, _) = rest.split_at_mut(1);
                let h_all_slice: &mut [f32] = &mut *fourth[0];

                let base = slice.start;
                let w_start = base;
                let u_start = w_start + d * d;
                let b_start = u_start + d;

                // h_prev в row-major (batch × d) — маленький временный вектор,
                // обновляется по шагам времени t.
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

                            // Column-major: (r, t, j) → j * (batch * seq) + (r * seq + t).
                            let h_idx = j * (batch * seq) + (r * seq + t);
                            h_all_slice[h_idx] = h;

                            h_prev[r * d + j] = h;

                            let out_idx = (t * d + j) * batch + r;
                            y[out_idx] = h;
                        }
                    }
                }
            });

        BufferedContext::IndRNN {
            input: input.clone(),
            h_all,
        }
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
        let (input_handle, h_all_handle) = match bc {
            BufferedContext::IndRNN { input, h_all } => (input, h_all),
            _ => panic!("Expected IndRNN Buffered context"),
        };

        let batch = grad_output.rows();
        let d = self.input_dim;
        let seq = self.seq_len;

        debug_assert_eq!(grad_output.cols(), seq * d);
        debug_assert_eq!(grad_input.rows(), batch);
        debug_assert_eq!(grad_input.cols(), seq * d);
        debug_assert_eq!(input_handle.rows(), batch);
        debug_assert_eq!(input_handle.cols(), seq * d);
        debug_assert_eq!(h_all_handle.rows(), batch * seq);
        debug_assert_eq!(h_all_handle.cols(), d);
        debug_assert!(
            slice.start + self.param_len() <= params.rows() * params.cols(),
            "IndRNN backward: parameter slice out of bounds"
        );
        debug_assert!(
            slice.start + self.param_len() <= grad_params.rows() * grad_params.cols(),
            "IndRNN backward: grad parameter slice out of bounds"
        );

        let ids = [
            input_handle.id(),
            h_all_handle.id(),
            grad_output.id(),
            grad_input.id(),
            params.id(),
            grad_params.id(),
        ];

        input_handle
            .memory()
            .write()
            .unwrap()
            .with_cpu_slices_mut(&ids, |slices| {
                let (first, rest) = slices.split_at_mut(1);
                let x: &[f32] = &*first[0];

                let (second, rest) = rest.split_at_mut(1);
                let h_all: &[f32] = &*second[0];

                let (third, rest) = rest.split_at_mut(1);
                let go: &[f32] = &*third[0];

                let (fourth, rest) = rest.split_at_mut(1);
                let gi: &mut [f32] = &mut *fourth[0];

                let (fifth, rest) = rest.split_at_mut(1);
                let p: &[f32] = &*fifth[0];

                let (sixth, _) = rest.split_at_mut(1);
                let gp: &mut [f32] = &mut *sixth[0];

                let base = slice.start;
                let w_start = base;
                let u_start = w_start + d * d;
                let b_start = u_start + d;

                // Обнуление градиентов.
                for i in 0..self.param_len() {
                    gp[base + i] = 0.0;
                }
                for i in 0..(batch * seq * d) {
                    gi[i] = 0.0;
                }

                let mut grad_w = vec![0.0f32; d * d];
                let mut grad_u = vec![0.0f32; d];
                let mut grad_b = vec![0.0f32; d];

                // delta_next в row-major (batch × d) — градиент, приходящий
                // из будущего шага.
                let mut delta_next = vec![0.0f32; batch * d];

                for t in (0..seq).rev() {
                    for r in 0..batch {
                        for j in 0..d {
                            let h_idx = j * (batch * seq) + (r * seq + t);
                            let h_t = h_all[h_idx];

                            let grad_out_t = go[(t * d + j) * batch + r];

                            let d_l_dh = grad_out_t
                                + p[u_start + j] * delta_next[r * d + j];

                            let d_relu = if h_t > 0.0 { 1.0 } else { 0.0 };
                            let delta_t = d_l_dh * d_relu;

                            grad_b[j] += delta_t;

                            if t > 0 {
                                let h_prev_idx =
                                    j * (batch * seq) + (r * seq + t - 1);
                                let h_prev = h_all[h_prev_idx];
                                grad_u[j] += delta_t * h_prev;
                            }

                            for i in 0..d {
                                let x_t_i = x[(t * d + i) * batch + r];
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