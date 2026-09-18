// src/layers/spectral_norm_linear/cpu/mod.rs

use crate::compute_manager::core::dynamic_context::DynamicContext;
use crate::compute_manager::operators_v2::memory_v2::buffer::{MatrixBufferHandle, TempMatrixPool};
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
        pool: &mut TempMatrixPool,
    ) -> BufferedContext {
        let batch = input.rows();
        let in_feat = self.in_features;
        let out_feat = self.out_features;

        debug_assert_eq!(input.cols(), in_feat);
        debug_assert_eq!(output.cols(), out_feat);
        debug_assert!(
            slice.start + self.param_len() <= params.rows() * params.cols(),
            "SpectrallyNormalizedLinear: parameter slice out of bounds"
        );

        // ------------------------------------------------------------------
        // Шаг 1. Читаем все параметры слоя одним коротким read_range.
        //         Удерживать блокировку MemoryExecutor долго нельзя — от
        //         неё зависит весь параллельный доступ к пулу.
        // ------------------------------------------------------------------
        let param_len = self.param_len();
        let p_vec = params.read_range(slice.start, param_len);

        let w_len = in_feat * out_feat;
        let w_slice = &p_vec[0..w_len];
        let bias_start_in_pvec = w_len;
        let scale_idx_in_pvec = bias_start_in_pvec + out_feat;
        let scale = p_vec[scale_idx_in_pvec];

        // ------------------------------------------------------------------
        // Шаг 2. Создаём per-chunk state-буферы степенного метода.
        //
        //   u_state     : (in_features × 1)
        //   v_state     : (out_features × 1)
        //   sigma_state : (1 × 1)
        // ------------------------------------------------------------------
        let u_state = pool.acquire(in_feat, 1);
        let v_state = pool.acquire(out_feat, 1);
        let sigma_state = pool.acquire(1, 1);

        // ------------------------------------------------------------------
        // Шаг 3. Power iteration (per-chunk, старт с u = v = 1).
        //
        //   v ← W · u ; v ← v / ‖v‖
        //   u ← Wᵀ · v ; u ← u / ‖u‖
        //   σ ← uᵀ · W · v
        // ------------------------------------------------------------------
        let mut u = vec![1.0f32; in_feat];
        let mut v = vec![1.0f32; out_feat];

        // v = W · u.
        for i in 0..out_feat {
            let mut sum = 0.0f32;
            for j in 0..in_feat {
                sum += w_slice[i * in_feat + j] * u[j];
            }
            v[i] = sum;
        }

        // v ← v / ‖v‖.
        let norm_v: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_v > 1e-12 {
            for vi in v.iter_mut() {
                *vi /= norm_v;
            }
        }

        // u = Wᵀ · v.
        for j in 0..in_feat {
            let mut sum = 0.0f32;
            for i in 0..out_feat {
                sum += w_slice[i * in_feat + j] * v[i];
            }
            u[j] = sum;
        }

        // u ← u / ‖u‖.
        let norm_u: f32 = u.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_u > 1e-12 {
            for ui in u.iter_mut() {
                *ui /= norm_u;
            }
        }

        // σ = uᵀ · W · v.
        let mut sigma = 0.0f32;
        for i in 0..out_feat {
            for j in 0..in_feat {
                sigma += u[j] * w_slice[i * in_feat + j] * v[i];
            }
        }
        sigma = sigma.abs().max(1e-12);

        // ------------------------------------------------------------------
        // Шаг 4. Сохраняем u, v, σ в per-chunk буферы.
        //
        //         Эти короткие write_range не пересекаются по времени с
        //         основным forward, поэтому MemoryExecutor освобождается
        //         между операциями.
        // ------------------------------------------------------------------
        u_state.write_range(0, &u);
        v_state.write_range(0, &v);
        sigma_state.write_range(0, &[sigma]);

        // ------------------------------------------------------------------
        // Шаг 5. Основной forward: y = (scale / σ) · W · x + b.
        //
        //         В этом блоке обращаемся только к input/output/params.
        //         State-буферы уже записаны, к ним больше не прикасаемся.
        // ------------------------------------------------------------------
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

                let (third, _) = rest.split_at_mut(1);
                let p: &[f32] = &*third[0];

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

        // ------------------------------------------------------------------
        // Шаг 6. Возвращаем контекст с дескрипторами state-буферов.
        // ------------------------------------------------------------------
        BufferedContext::SpectralNormLinear {
            input: input.clone(),
            u_state,
            v_state,
            sigma_state,
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
        // ------------------------------------------------------------------
        // Шаг 1. Разбор контекста: нужны input и sigma. u/v в backward
        //         не используются (классический Spectral Normalization
        //         трактует u и v как detached estimators).
        // ------------------------------------------------------------------
        let DynamicContext::Buffered(bc) = ctx;
        let (input_handle, sigma_state) = match bc {
            BufferedContext::SpectralNormLinear {
                input, sigma_state, ..
            } => (input, sigma_state),
            _ => panic!("Expected SpectralNormLinear Buffered context"),
        };

        let batch = grad_output.rows();
        let in_feat = self.in_features;
        let out_feat = self.out_features;

        debug_assert_eq!(input_handle.cols(), in_feat);
        debug_assert_eq!(grad_output.cols(), out_feat);
        debug_assert_eq!(grad_input.cols(), in_feat);
        debug_assert_eq!(grad_input.rows(), batch);
        debug_assert!(
            slice.start + self.param_len() <= params.rows() * params.cols(),
            "SpectrallyNormalizedLinear backward: parameter slice out of bounds"
        );
        debug_assert!(
            slice.start + self.param_len() <= grad_params.rows() * grad_params.cols(),
            "SpectrallyNormalizedLinear backward: grad parameter slice out of bounds"
        );

        // ------------------------------------------------------------------
        // Шаг 2. Читаем sigma из state-буфера.
        // ------------------------------------------------------------------
        let sigma = {
            let s = sigma_state.read_range(0, 1);
            s[0]
        };
        debug_assert!(sigma.is_finite() && sigma > 0.0);

        // ------------------------------------------------------------------
        // Шаг 3. Основной backward.
        //
        //   grad_b[i]      = Σ_r go[i,r]
        //   grad_scale     = Σ_{r,i} go[i,r] · (W x)[i,r] / σ
        //   grad_W[i,j]    = Σ_r (scale/σ) · go[i,r] · x[j,r]
        //   gi[j,r]        = Σ_i (scale/σ) · W[i,j] · go[i,r]
        // ------------------------------------------------------------------
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

                let (fifth, _) = rest.split_at_mut(1);
                let gp: &mut [f32] = &mut *fifth[0];

                let base = slice.start;
                let w_start = base;
                let b_start = w_start + in_feat * out_feat;
                let scale_idx = b_start + out_feat;
                let scale = p[scale_idx];
                let effective_scale = scale / sigma;

                // Обнуляем градиенты параметров.
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

                        // grad_scale += gout * (W x)_i / σ.
                        let mut wx = 0.0f32;
                        for j in 0..in_feat {
                            wx += p[w_start + i * in_feat + j] * x[j * batch + r];
                        }
                        grad_scale += gout * wx / sigma;

                        // grad_W[i,j] += effective_scale · gout · x[j,r].
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