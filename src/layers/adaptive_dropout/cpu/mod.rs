// src/layers/adaptive_dropout/cpu/mod.rs

use rand::Rng;
use crate::compute_manager::graph::types::DynamicContext;
use crate::compute_manager::matrix_buffer::MatrixBufferHandle;
use crate::layers::buffered_context::BufferedContext;
use crate::layers::UniversalLayerBuffered;
use crate::model_plan::param_store::ParamSlice;

use super::super::adaptive_dropout::AdaptiveDropout;

impl UniversalLayerBuffered for AdaptiveDropout {
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
        debug_assert!(slice.start + self.param_len() <= params.rows() * params.cols());

        // Читаем параметры (θ, T)
        let (theta, temp) = {
            let p_guard = params.read();
            let p = p_guard.as_slice().unwrap();
            let theta_start = slice.start;
            let temp_start = theta_start + self.features;
            (
                p[theta_start..theta_start + self.features].to_vec(),
                p[temp_start..temp_start + self.features].to_vec(),
            )
        };

        let eps = 1e-6;
        let mut rng = rand::thread_rng();
        let total = rows * cols;

        // Генерируем маску и вероятности
        let mut mask = vec![0.0f32; total];
        let mut probs = vec![0.0f32; total];

        {
            let input_guard = input.read();
            let x = input_guard.as_slice().unwrap();

            for c in 0..cols {
                let theta_c = theta[c];
                let temp_c = temp[c].abs() + eps; // гарантируем положительность
                for r in 0..rows {
                    let idx = c * rows + r;
                    let x_val = x[idx].abs();
                    // p = sigmoid( (|x| - θ) / T )
                    let p = 1.0 / (1.0 + (-(x_val - theta_c) / temp_c).exp());
                    probs[idx] = p;
                    // z ~ Bernoulli(p)
                    let keep = rng.gen::<f32>() < p;
                    mask[idx] = if keep { 1.0 } else { 0.0 };
                }
            }
        }

        // Сохраняем маску для обратного прохода
        *self.mask.lock().unwrap() = Some(mask.clone());

        // Вычисляем выход: y = x * z / (p + eps)
        {
            let input_guard = input.read();
            let x = input_guard.as_slice().unwrap();
            let mut output_guard = output.write();
            let y = output_guard.as_slice_mut().unwrap();
            for i in 0..total {
                y[i] = x[i] * mask[i] / (probs[i] + eps);
            }
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
            BufferedContext::AdaptiveDropout { input } => input,
            _ => panic!("Expected AdaptiveDropout context"),
        };

        let mask = self.mask.lock().unwrap().take()
            .expect("AdaptiveDropout backward called without forward mask");

        let rows = grad_output.rows();
        let cols = grad_output.cols();
        let total = rows * cols;
        debug_assert_eq!(cols, self.features);
        debug_assert_eq!(rows, input_handle.rows());

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

                let theta_start = slice.start;
                let temp_start = theta_start + self.features;
                let eps = 1e-6;

                let mut grad_theta = vec![0.0f32; self.features];
                let mut grad_temp = vec![0.0f32; self.features];

                for c in 0..cols {
                    let theta_c = p[theta_start + c];
                    let temp_c = p[temp_start + c].abs() + eps;

                    let mut d_theta_acc = 0.0;
                    let mut d_temp_acc = 0.0;

                    for r in 0..rows {
                        let idx = c * rows + r;
                        let x_val = x[idx];
                        let gout = go[idx];
                        let prob = 1.0 / (1.0 + (-(x_val.abs() - theta_c) / temp_c).exp());
                        let z = mask[idx];

                        // Градиент по входу: dL/dx = gout * z / (prob + eps)
                        gi[idx] = gout * z / (prob + eps);

                        // Производные sigmoid по параметрам
                        let dsig_darg = prob * (1.0 - prob);
                        // d(prob)/d(theta) = - dsig_darg / temp_c
                        let dprob_dtheta = -dsig_darg / temp_c;
                        // d(prob)/d(temp) = - dsig_darg * (x_abs - theta_c) / (temp_c^2)
                        let dprob_dtemp = -dsig_darg * (x_val.abs() - theta_c) / (temp_c * temp_c);

                        // Градиент по θ: dL/dθ = sum_r dL/dy * dy/dprob * dprob/dθ
                        // dy/dprob = - x * z / (prob + eps)^2
                        let dy_dprob = -x_val * z / ((prob + eps) * (prob + eps));
                        d_theta_acc += gout * dy_dprob * dprob_dtheta;
                        d_temp_acc += gout * dy_dprob * dprob_dtemp;
                    }

                    grad_theta[c] = d_theta_acc;
                    grad_temp[c] = d_temp_acc;
                }

                // Записываем градиенты параметров
                for c in 0..self.features {
                    gp[theta_start + c] = grad_theta[c];
                    gp[temp_start + c] = grad_temp[c];
                }
            });
    }

    fn param_len(&self) -> usize {
        2 * self.features
    }

    fn input_features(&self) -> usize {
        self.features
    }

    fn output_features(&self) -> usize {
        self.features
    }
}