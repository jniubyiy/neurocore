// src/layers/spectral_norm_linear/cpu/mod.rs

use crate::compute_manager::graph::types::DynamicContext;
use crate::compute_manager::matrix_buffer::MatrixBufferHandle;
use crate::layers::buffered_context::BufferedContext;
use crate::layers::UniversalLayerBuffered;
use crate::model_plan::param_store::ParamSlice;

use super::super::spectral_norm_linear::SpectrallyNormalizedLinear;

impl UniversalLayerBuffered for SpectrallyNormalizedLinear {
    fn forward_buffered(
        &self,
        input: &MatrixBufferHandle,
        output: &MatrixBufferHandle,
        params: &MatrixBufferHandle,
        slice: &ParamSlice,
    ) {
        let batch = input.rows();
        let in_feat = self.in_features;
        let out_feat = self.out_features;
        debug_assert_eq!(input.cols(), in_feat);
        debug_assert_eq!(output.cols(), out_feat);
        debug_assert!(slice.start + self.param_len() <= params.rows() * params.cols());

        // ============ 1. Обновление u, v, sigma (степенной метод) ============
        // Читаем веса W и scale из общего буфера параметров для обновления u, v.
        // Работаем с state под write(), затем сразу отпускаем.
        let scale = {
            let p_guard = params.read();
            let p = p_guard.as_slice().unwrap();
            let base = slice.start;
            let w_start = base;
            let scale_idx = w_start + in_feat * out_feat + out_feat;

            let mut guard = self.state.write().unwrap();
            let state = &mut *guard;

            if !state.initialized {
                state.u.fill(1.0);
                state.v.fill(1.0);
                state.initialized = true;
            }

            // v = W u
            for i in 0..out_feat {
                let mut sum = 0.0f32;
                for j in 0..in_feat {
                    sum += p[w_start + i * in_feat + j] * state.u[j];
                }
                state.v[i] = sum;
            }

            // Нормализация v.
            let norm_v: f32 = state.v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm_v > 1e-12 {
                for vi in state.v.iter_mut() {
                    *vi /= norm_v;
                }
            }

            // u = W^T v
            for j in 0..in_feat {
                let mut sum = 0.0f32;
                for i in 0..out_feat {
                    sum += p[w_start + i * in_feat + j] * state.v[i];
                }
                state.u[j] = sum;
            }

            // Нормализация u.
            let norm_u: f32 = state.u.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm_u > 1e-12 {
                for ui in state.u.iter_mut() {
                    *ui /= norm_u;
                }
            }

            // sigma = u^T W v
            let mut sigma = 0.0f32;
            for i in 0..out_feat {
                for j in 0..in_feat {
                    sigma += state.u[j] * p[w_start + i * in_feat + j] * state.v[i];
                }
            }
            sigma = sigma.abs().max(1e-12);
            state.last_sigma = sigma;

            p[scale_idx]
        };

        // ============ 2. Прямой проход: y = W_sn x + b ============
        let sigma = self.get_last_sigma();
        let effective_scale = scale / sigma;

        let ids = [input.id(), output.id(), params.id()];
        input
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
                let b_start = w_start + in_feat * out_feat;

                for r in 0..batch {
                    for i in 0..out_feat {
                        let mut sum = p[b_start + i];
                        for j in 0..in_feat {
                            sum += effective_scale
                                * p[w_start + i * in_feat + j]
                                * x[j * batch + r];
                        }
                        y[i * batch + r] = sum;
                    }
                }
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
        let input_handle = match bc {
            BufferedContext::SpectralNormLinear { input } => input,
            _ => panic!("Expected SpectralNormLinear context"),
        };

        let batch = grad_output.rows();
        let in_feat = self.in_features;
        let out_feat = self.out_features;
        debug_assert_eq!(input_handle.cols(), in_feat);
        debug_assert_eq!(grad_output.cols(), out_feat);
        debug_assert_eq!(grad_input.cols(), in_feat);
        debug_assert_eq!(grad_input.rows(), batch);
        debug_assert!(slice.start + self.param_len() <= params.rows() * params.cols());
        debug_assert!(
            slice.start + self.param_len() <= grad_params.rows() * grad_params.cols()
        );

        // Читаем sigma под read() — она была сохранена в forward.
        let sigma = self.get_last_sigma();

        let ids = [
            input_handle.id(),
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
                let go: &[f32] = &*second[0];
                let (third, rest) = rest.split_at_mut(1);
                let gi: &mut [f32] = &mut *third[0];
                let (fourth, rest) = rest.split_at_mut(1);
                let p: &[f32] = &*fourth[0];
                let gp: &mut [f32] = &mut *rest[0];

                let base = slice.start;
                let w_start = base;
                let b_start = w_start + in_feat * out_feat;
                let scale_idx = b_start + out_feat;
                let scale = p[scale_idx];
                let effective_scale = scale / sigma;

                // Инициализация градиентов параметров.
                for i in 0..self.param_len() {
                    gp[base + i] = 0.0;
                }

                let mut grad_w = vec![0.0f32; in_feat * out_feat];
                let mut grad_b = vec![0.0f32; out_feat];
                let mut grad_scale = 0.0f32;

                // Градиенты по bias, scale и W.
                for r in 0..batch {
                    for i in 0..out_feat {
                        let gout = go[i * batch + r];
                        grad_b[i] += gout;

                        // grad_scale += gout * (W x)_i / sigma
                        let mut wx = 0.0f32;
                        for j in 0..in_feat {
                            wx += p[w_start + i * in_feat + j] * x[j * batch + r];
                        }
                        grad_scale += gout * wx / sigma;

                        // grad_W[i,j] += effective_scale * gout * x[j,r]
                        for j in 0..in_feat {
                            grad_w[i * in_feat + j] +=
                                effective_scale * gout * x[j * batch + r];
                        }
                    }
                }

                // Градиент по входу.
                for j in 0..in_feat {
                    for r in 0..batch {
                        let mut sum = 0.0f32;
                        for i in 0..out_feat {
                            sum += effective_scale
                                * p[w_start + i * in_feat + j]
                                * go[i * batch + r];
                        }
                        gi[j * batch + r] = sum;
                    }
                }

                // Запись градиентов параметров.
                for i in 0..out_feat {
                    gp[b_start + i] = grad_b[i];
                }
                for idx in 0..(in_feat * out_feat) {
                    gp[w_start + idx] = grad_w[idx];
                }
                gp[scale_idx] = grad_scale;
            });
    }

    fn param_len(&self) -> usize {
        self.in_features * self.out_features + self.out_features + 1
    }

    fn input_features(&self) -> usize {
        self.in_features
    }

    fn output_features(&self) -> usize {
        self.out_features
    }
}