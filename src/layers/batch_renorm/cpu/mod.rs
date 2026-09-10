// src/layers/batch_renorm/cpu/mod.rs

use crate::compute_manager::graph::types::DynamicContext;
use crate::compute_manager::matrix_buffer::MatrixBufferHandle;
use crate::layers::buffered_context::BufferedContext;
use crate::layers::UniversalLayerBuffered;
use crate::model_plan::param_store::ParamSlice;

use super::super::batch_renorm::BatchRenorm1d;

impl UniversalLayerBuffered for BatchRenorm1d {
    fn forward_buffered(
        &self,
        input: &MatrixBufferHandle,
        output: &MatrixBufferHandle,
        params: &MatrixBufferHandle,
        slice: &ParamSlice,
    ) {
        let rows = input.rows();
        let cols = input.cols();
        debug_assert_eq!(cols, self.features);
        debug_assert!(
            slice.start + self.param_len() <= params.rows() * params.cols(),
            "BatchRenorm1d: parameter slice out of bounds"
        );

        let f = self.features;
        let eps = self.eps;
        let momentum = self.momentum;

        // Локальные копии статистик и режима — забираем под одним read().
        let (training, running_mean_local, running_var_local) = {
            let state = self.state.read().unwrap();
            (
                state.training,
                state.running_mean.clone(),
                state.running_var.clone(),
            )
        };

        let ids = [input.id(), output.id(), params.id()];

        // Возвращаем новые running-статистики, если обучаемся.
        let new_running: Option<(Vec<f32>, Vec<f32>)> = input
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
                let gamma_start = base;
                let beta_start = gamma_start + f;
                let r_start = beta_start + f;
                let d_start = r_start + f;

                // ============ 1. Статистики ============
                let (mean, var): (Vec<f32>, Vec<f32>) = if training {
                    let mut batch_mean = vec![0.0f32; f];
                    let mut batch_var = vec![0.0f32; f];
                    for c in 0..cols {
                        let mut sum = 0.0f32;
                        let mut sum_sq = 0.0f32;
                        for r in 0..rows {
                            let idx = c * rows + r;
                            let v = x[idx];
                            sum += v;
                            sum_sq += v * v;
                        }
                        let mean_c = sum / rows as f32;
                        let var_c = (sum_sq / rows as f32) - mean_c * mean_c;
                        batch_mean[c] = mean_c;
                        batch_var[c] = var_c.max(0.0f32);
                    }
                    (batch_mean, batch_var)
                } else {
                    (running_mean_local.clone(), running_var_local.clone())
                };

                // ============ 2. Обновление running-статистик (если обучаемся) ============
                let new_stats = if training {
                    let mut nm = vec![0.0f32; f];
                    let mut nv = vec![0.0f32; f];
                    for c in 0..f {
                        nm[c] = (1.0 - momentum) * running_mean_local[c]
                            + momentum * mean[c];
                        nv[c] = (1.0 - momentum) * running_var_local[c]
                            + momentum * var[c];
                    }
                    Some((nm, nv))
                } else {
                    None
                };

                // ============ 3. Прямой проход ============
                for c in 0..cols {
                    let gamma = p[gamma_start + c];
                    let beta = p[beta_start + c];
                    let r_par = p[r_start + c];
                    let d_par = p[d_start + c];
                    let mean_c = mean[c];
                    let var_c = var[c];
                    let inv_std = 1.0 / (var_c + eps).sqrt();
                    for row in 0..rows {
                        let idx = c * rows + row;
                        let x_hat = (x[idx] - mean_c) * inv_std;
                        y[idx] = x_hat * r_par * gamma + d_par * gamma + beta;
                    }
                }

                new_stats
            });

        // ============ 4. Запись новых running-статистик ============
        if let Some((nm, nv)) = new_running {
            let mut state = self.state.write().unwrap();
            state.running_mean = nm;
            state.running_var = nv;
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
        let input_handle = match bc {
            BufferedContext::BatchRenorm { input, .. } => input,
            _ => panic!("Expected BatchRenorm context"),
        };

        let rows = grad_output.rows();
        let cols = grad_output.cols();
        debug_assert_eq!(cols, self.features);
        debug_assert_eq!(rows, input_handle.rows());
        debug_assert!(
            slice.start + self.param_len() <= params.rows() * params.cols(),
            "BatchRenorm1d backward: parameter slice out of bounds"
        );
        debug_assert!(
            slice.start + self.param_len() <= grad_params.rows() * grad_params.cols(),
            "BatchRenorm1d backward: grad parameter slice out of bounds"
        );

        let f = self.features;
        let eps = self.eps;

        // Читаем режим и статистики под одним read().
        let (training, running_mean, running_var) = {
            let state = self.state.read().unwrap();
            (
                state.training,
                state.running_mean.clone(),
                state.running_var.clone(),
            )
        };

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
                let gamma_start = base;
                let beta_start = gamma_start + f;
                let r_start = beta_start + f;
                let d_start = r_start + f;

                // Обнуляем градиенты параметров.
                for i in 0..(4 * f) {
                    gp[base + i] = 0.0f32;
                }

                // ============ 1. Пересчитываем статистики ============
                let (mean, var): (Vec<f32>, Vec<f32>) = if training {
                    let mut batch_mean = vec![0.0f32; f];
                    let mut batch_var = vec![0.0f32; f];
                    for c in 0..cols {
                        let mut sum = 0.0f32;
                        let mut sum_sq = 0.0f32;
                        for r in 0..rows {
                            let idx = c * rows + r;
                            let v = x[idx];
                            sum += v;
                            sum_sq += v * v;
                        }
                        let mean_c = sum / rows as f32;
                        let var_c = (sum_sq / rows as f32) - mean_c * mean_c;
                        batch_mean[c] = mean_c;
                        batch_var[c] = var_c.max(0.0f32);
                    }
                    (batch_mean, batch_var)
                } else {
                    (running_mean.clone(), running_var.clone())
                };

                let mut grad_gamma = vec![0.0f32; f];
                let mut grad_beta = vec![0.0f32; f];
                let mut grad_r = vec![0.0f32; f];
                let mut grad_d = vec![0.0f32; f];

                // ============ 2. Градиенты по параметрам ============
                for c in 0..cols {
                    let gamma = p[gamma_start + c];
                    let r_par = p[r_start + c];
                    let d_par = p[d_start + c];
                    let mean_c = mean[c];
                    let var_c = var[c];
                    let inv_std = 1.0 / (var_c + eps).sqrt();

                    let mut sum_gamma_r = 0.0f32;
                    let mut sum_gamma_r_xhat = 0.0f32;

                    for row in 0..rows {
                        let idx = c * rows + row;
                        let x_val = x[idx];
                        let gout = go[idx];
                        let x_hat = (x_val - mean_c) * inv_std;

                        grad_gamma[c] += gout * (x_hat * r_par + d_par);
                        grad_beta[c] += gout;
                        grad_r[c] += gout * gamma * x_hat;
                        grad_d[c] += gout * gamma;

                        if training {
                            let gy = gout * gamma * r_par;
                            sum_gamma_r += gy;
                            sum_gamma_r_xhat += gy * x_hat;
                        }
                    }

                    // ============ 3. Градиент по входу ============
                    for row in 0..rows {
                        let idx = c * rows + row;
                        let gout = go[idx];
                        let x_hat = (x[idx] - mean_c) * inv_std;

                        if training {
                            let n = rows as f32;
                            let term1 = gout * gamma * r_par * inv_std;
                            let term2 = sum_gamma_r / n;
                            let term3 =
                                x_hat * sum_gamma_r_xhat / (n * (var_c + eps).sqrt());
                            gi[idx] = term1 - term2 - term3;
                        } else {
                            gi[idx] = gout * gamma * r_par * inv_std;
                        }
                    }
                }

                // ============ 4. Запись градиентов параметров ============
                for c in 0..f {
                    gp[gamma_start + c] = grad_gamma[c];
                    gp[beta_start + c] = grad_beta[c];
                    gp[r_start + c] = grad_r[c];
                    gp[d_start + c] = grad_d[c];
                }
            });
    }

    fn param_len(&self) -> usize {
        4 * self.features
    }

    fn input_features(&self) -> usize {
        self.features
    }

    fn output_features(&self) -> usize {
        self.features
    }
}