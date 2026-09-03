// src/layers/multi_resolution_kan_linear/cpu/mod.rs

use crate::compute_manager::graph::types::DynamicContext;
use crate::compute_manager::matrix_buffer::MatrixBufferHandle;
use crate::layers::buffered_context::BufferedContext;
use crate::layers::UniversalLayerBuffered;
use crate::model_plan::param_store::ParamSlice;

use super::super::multi_resolution_kan_linear::MultiResolutionKANLinear;

const COARSE_GRID: usize = 4;
const FINE_GRID: usize = 8;

impl UniversalLayerBuffered for MultiResolutionKANLinear {
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
            let coarse_offset = base;
            let fine_offset = coarse_offset + in_feat * out_feat * COARSE_GRID;
            let bias_offset = fine_offset + in_feat * out_feat * FINE_GRID;

            for r in 0..batch {
                for j in 0..out_feat {
                    let mut sum = p[bias_offset + j];
                    for i in 0..in_feat {
                        let x_val = x[i * batch + r].clamp(-1.0, 1.0);

                        // Грубая сетка
                        let coarse_base = coarse_offset + (j * in_feat + i) * COARSE_GRID;
                        let coarse_val = linear_interpolate(x_val, COARSE_GRID, &p[coarse_base..coarse_base + COARSE_GRID]);
                        // Точная сетка
                        let fine_base = fine_offset + (j * in_feat + i) * FINE_GRID;
                        let fine_val = linear_interpolate(x_val, FINE_GRID, &p[fine_base..fine_base + FINE_GRID]);

                        sum += coarse_val + fine_val;
                    }
                    y[j * batch + r] = sum;
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
            BufferedContext::MultiResolutionKANLinear { input } => input,
            _ => panic!("Expected MultiResolutionKANLinear context"),
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
                let coarse_offset = base;
                let fine_offset = coarse_offset + in_feat * out_feat * COARSE_GRID;
                let bias_offset = fine_offset + in_feat * out_feat * FINE_GRID;

                // Инициализируем градиенты параметров нулями
                let mut grad_coarse = vec![0.0f32; in_feat * out_feat * COARSE_GRID];
                let mut grad_fine = vec![0.0f32; in_feat * out_feat * FINE_GRID];
                let mut grad_bias = vec![0.0f32; out_feat];

                for r in 0..batch {
                    for j in 0..out_feat {
                        let gout = go[j * batch + r];
                        grad_bias[j] += gout;
                        for i in 0..in_feat {
                            let x_val = x[i * batch + r].clamp(-1.0, 1.0);

                            // Грубая сетка
                            let coarse_base = coarse_offset + (j * in_feat + i) * COARSE_GRID;
                            let (c_val, c_deriv, c_indices, c_weights) = linear_interpolate_deriv(
                                x_val, COARSE_GRID, &p[coarse_base..coarse_base + COARSE_GRID]
                            );
                            // Точная сетка
                            let fine_base = fine_offset + (j * in_feat + i) * FINE_GRID;
                            let (f_val, f_deriv, f_indices, f_weights) = linear_interpolate_deriv(
                                x_val, FINE_GRID, &p[fine_base..fine_base + FINE_GRID]
                            );

                            // Градиент по входу
                            gi[i * batch + r] += gout * (c_deriv + f_deriv);

                            // Градиенты по коэффициентам
                            for k in 0..2 {
                                let idx_c = c_indices[k];
                                let w_c = c_weights[k];
                                grad_coarse[(j * in_feat + i) * COARSE_GRID + idx_c] += gout * w_c;
                            }
                            for k in 0..2 {
                                let idx_f = f_indices[k];
                                let w_f = f_weights[k];
                                grad_fine[(j * in_feat + i) * FINE_GRID + idx_f] += gout * w_f;
                            }
                        }
                    }
                }

                // Записываем градиенты параметров
                for j in 0..out_feat {
                    gp[bias_offset + j] = grad_bias[j];
                }
                for idx in 0..(in_feat * out_feat * COARSE_GRID) {
                    gp[coarse_offset + idx] = grad_coarse[idx];
                }
                for idx in 0..(in_feat * out_feat * FINE_GRID) {
                    gp[fine_offset + idx] = grad_fine[idx];
                }
            });
    }

    fn param_len(&self) -> usize {
        self.in_features * self.out_features * (COARSE_GRID + FINE_GRID) + self.out_features
    }

    fn input_features(&self) -> usize {
        self.in_features
    }

    fn output_features(&self) -> usize {
        self.out_features
    }
}

/// Линейная интерполяция по равномерной сетке на [-1, 1].
/// Возвращает интерполированное значение.
fn linear_interpolate(x: f32, grid_size: usize, coeffs: &[f32]) -> f32 {
    let (val, _, _, _) = linear_interpolate_deriv(x, grid_size, coeffs);
    val
}

/// Линейная интерполяция с возвратом значения, производной и информации для градиентов.
fn linear_interpolate_deriv(x: f32, grid_size: usize, coeffs: &[f32]) -> (f32, f32, [usize; 2], [f32; 2]) {
    // Нормализуем x в [0,1]
    let t = (x + 1.0) * 0.5;
    let scaled = t * (grid_size - 1) as f32;
    let idx0 = scaled.floor() as usize;
    let idx1 = (idx0 + 1).min(grid_size - 1);
    let frac = scaled - idx0 as f32;
    let w0 = 1.0 - frac;
    let w1 = frac;
    let val = w0 * coeffs[idx0] + w1 * coeffs[idx1];
    // Производная по x с учётом масштабирования: d/dx = d/dt * dt/dx = d/dt * 0.5
    // d/dt ≈ (coeffs[idx1] - coeffs[idx0]) * (grid_size-1), но с учётом нормализации.
    let deriv = (coeffs[idx1] - coeffs[idx0]) * (grid_size - 1) as f32 * 0.5;
    (val, deriv, [idx0, idx1], [w0, w1])
}