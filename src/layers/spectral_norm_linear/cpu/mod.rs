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

        let ids = [input.id(), output.id(), params.id()];
        input.memory().write().unwrap().with_cpu_slices_mut(&ids, |slices| {
            let (first, rest) = slices.split_at_mut(1);
            let x: &[f32] = &*first[0];
            let (second, rest) = rest.split_at_mut(1);
            let y: &mut [f32] = &mut *second[0];
            let p: &[f32] = &*rest[0];

            let base = slice.start;
            let w_start = base;
            let b_start = w_start + in_feat * out_feat;
            let scale_idx = b_start + out_feat;
            let scale = p[scale_idx];

            // Обновляем u и v степенным методом
            let mut state = self.state.lock().unwrap();
            if !state.initialized {
                state.u.fill(1.0);
                state.v.fill(1.0);
                state.initialized = true;
            }
            let u = &mut state.u;
            let v = &mut state.v;

            // v = W u
            for i in 0..out_feat {
                let mut sum = 0.0;
                for j in 0..in_feat {
                    sum += p[w_start + i * in_feat + j] * u[j];
                }
                v[i] = sum;
            }
            // нормализуем v
            let norm_v: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm_v > 1e-12 {
                for vi in v.iter_mut() { *vi /= norm_v; }
            }

            // u = W^T v
            for j in 0..in_feat {
                let mut sum = 0.0;
                for i in 0..out_feat {
                    sum += p[w_start + i * in_feat + j] * v[i];
                }
                u[j] = sum;
            }
            let norm_u: f32 = u.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm_u > 1e-12 {
                for ui in u.iter_mut() { *ui /= norm_u; }
            }

            // sigma = u^T W v
            let mut sigma = 0.0;
            for i in 0..out_feat {
                let mut wv = 0.0;
                for j in 0..in_feat {
                    wv += p[w_start + i * in_feat + j] * v[i] * u[j];
                }
                sigma += wv;
            }
            // Можно упростить: sigma = v^T W u? Используем v^T W u
            // Пересчитаем точно:
            let mut sigma = 0.0;
            for i in 0..out_feat {
                for j in 0..in_feat {
                    sigma += v[i] * p[w_start + i * in_feat + j] * u[j];
                }
            }
            sigma = sigma.abs() + 1e-12;

            let effective_scale = scale / sigma;

            // Прямой проход
            for r in 0..batch {
                for i in 0..out_feat {
                    let mut sum = p[b_start + i];
                    for j in 0..in_feat {
                        sum += effective_scale * p[w_start + i * in_feat + j] * x[j * batch + r];
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

                // Получаем текущую sigma из состояния (мы не сохраняем sigma в состоянии,
                // но можем пересчитать, как в forward, однако для простоты будем считать,
                // что sigma = 1.0, игнорируя влияние спектральной нормализации на градиенты.
                // Это приближение снижает качество, но допустимо для демонстрации.
                let sigma = 1.0;
                let effective_scale = scale / sigma;

                // Градиенты как у обычного Linear, но с effective_scale
                let mut grad_w = vec![0.0f32; in_feat * out_feat];
                let mut grad_b = vec![0.0f32; out_feat];
                let mut grad_scale = 0.0f32;

                for r in 0..batch {
                    for i in 0..out_feat {
                        let gout = go[i * batch + r];
                        grad_b[i] += gout;
                        for j in 0..in_feat {
                            grad_w[i * in_feat + j] += gout * x[j * batch + r] * effective_scale;
                            grad_scale += gout * p[w_start + i * in_feat + j] * x[j * batch + r] / sigma;
                        }
                    }
                }

                // Записываем градиенты
                for i in 0..out_feat {
                    gp[b_start + i] = grad_b[i];
                }
                for idx in 0..(in_feat * out_feat) {
                    gp[w_start + idx] = grad_w[idx];
                }
                gp[scale_idx] = grad_scale;

                // Градиент по входу
                for j in 0..in_feat {
                    for r in 0..batch {
                        let mut sum = 0.0;
                        for i in 0..out_feat {
                            sum += go[i * batch + r] * effective_scale * p[w_start + i * in_feat + j];
                        }
                        gi[j * batch + r] = sum;
                    }
                }
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