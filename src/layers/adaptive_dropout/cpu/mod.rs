// src/layers/adaptive_dropout/cpu/mod.rs

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::compute_manager::graph::types::DynamicContext;
use crate::compute_manager::matrix_buffer::{MatrixBufferHandle, TempMatrixPool};
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
        pool: &mut TempMatrixPool,
    ) -> BufferedContext {
        let rows = input.rows();
        let cols = input.cols();
        let total = rows * cols;

        debug_assert_eq!(cols, self.features);
        debug_assert!(
            slice.start + self.param_len() <= params.rows() * params.cols(),
            "AdaptiveDropout: parameter slice out of bounds"
        );

        // ------------------------------------------------------------------
        // Per-chunk state-буферы:
        //   mask: (total × 1) — бинарная маска z ∈ {0, 1}.
        //   arg:  (total × 1) — аргумент сигмоиды a = (|x| − θ) / T.
        //
        // Оба буфера создаются из пула, живут в контексте до конца backward,
        // затем освобождаются.
        // ------------------------------------------------------------------
        let mask_handle = pool.acquire(total, 1);
        let arg_handle = pool.acquire(total, 1);

        // ------------------------------------------------------------------
        // Считываем θ и T из общего буфера параметров.
        // ------------------------------------------------------------------
        let (theta, temp) = {
            let p_guard = params.read();
            let p = p_guard
                .as_slice()
                .expect("AdaptiveDropout: params must be CPU");
            let theta_start = slice.start;
            let temp_start = theta_start + self.features;
            (
                p[theta_start..theta_start + self.features].to_vec(),
                p[temp_start..temp_start + self.features].to_vec(),
            )
        };

        let eps = 1e-6;
        let mut rng = StdRng::seed_from_u64(self.seed);

        // ------------------------------------------------------------------
        // Основной проход: генерируем маску, пишем arg, вычисляем выход.
        // ------------------------------------------------------------------
        let ids = [
            input.id(),
            output.id(),
            mask_handle.id(),
            arg_handle.id(),
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
                let mask: &mut [f32] = &mut *third[0];

                let (fourth, _) = rest.split_at_mut(1);
                let arg: &mut [f32] = &mut *fourth[0];

                for c in 0..cols {
                    let theta_c = theta[c];
                    let temp_c = temp[c].abs() + eps;
                    for r in 0..rows {
                        let idx = c * rows + r;
                        let x_val = x[idx].abs();
                        let a = (x_val - theta_c) / temp_c;
                        let p_keep = 1.0 / (1.0 + (-a).exp());

                        // Генерация Bernoulli для маски.
                        let keep = rng.gen::<f32>() < p_keep;
                        let z = if keep { 1.0 } else { 0.0 };

                        // Сохраняем a и z для обратного прохода.
                        arg[idx] = a;
                        mask[idx] = z;

                        // Выход: y = x · z / (p + eps).
                        y[idx] = x[idx] * z / (p_keep + eps);
                    }
                }
            });

        BufferedContext::AdaptiveDropout {
            input: input.clone(),
            mask: mask_handle,
            arg: arg_handle,
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
        let (input_handle, mask_handle, arg_handle) = match bc {
            BufferedContext::AdaptiveDropout { input, mask, arg } => {
                (input, mask, arg)
            }
            _ => panic!("Expected AdaptiveDropout Buffered context"),
        };

        let rows = grad_output.rows();
        let cols = grad_output.cols();
        let total = rows * cols;

        debug_assert_eq!(cols, self.features);
        debug_assert_eq!(rows, input_handle.rows());
        debug_assert_eq!(total, mask_handle.rows() * mask_handle.cols());
        debug_assert_eq!(total, arg_handle.rows() * arg_handle.cols());

        debug_assert!(
            slice.start + self.param_len() <= params.rows() * params.cols(),
            "AdaptiveDropout backward: parameter slice out of bounds"
        );
        debug_assert!(
            slice.start + self.param_len() <= grad_params.rows() * grad_params.cols(),
            "AdaptiveDropout backward: grad parameter slice out of bounds"
        );

        let eps = 1e-6;

        let ids = [
            input_handle.id(),
            grad_output.id(),
            grad_input.id(),
            params.id(),
            grad_params.id(),
            mask_handle.id(),
            arg_handle.id(),
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

                let (fifth, rest) = rest.split_at_mut(1);
                let gp: &mut [f32] = &mut *fifth[0];

                let (sixth, rest) = rest.split_at_mut(1);
                let mask: &[f32] = &*sixth[0];

                let (seventh, _) = rest.split_at_mut(1);
                let arg: &[f32] = &*seventh[0];

                let theta_start = slice.start;
                let temp_start = theta_start + self.features;

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
                        let z = mask[idx];
                        let a = arg[idx];

                        // Восстанавливаем p_keep из arg — сигмоида.
                        let p_keep = 1.0 / (1.0 + (-a).exp());

                        // Градиент по входу.
                        gi[idx] = gout * z / (p_keep + eps);

                        // Производные сигмоиды по theta и T.
                        let dsig_darg = p_keep * (1.0 - p_keep);
                        let dprob_dtheta = -dsig_darg / temp_c;
                        let dprob_dtemp = -dsig_darg * (x_val.abs() - theta_c)
                            / (temp_c * temp_c);

                        // Производная выхода по вероятности удержания.
                        let dy_dprob =
                            -x_val * z / ((p_keep + eps) * (p_keep + eps));

                        d_theta_acc += gout * dy_dprob * dprob_dtheta;
                        d_temp_acc += gout * dy_dprob * dprob_dtemp;
                    }

                    grad_theta[c] = d_theta_acc;
                    grad_temp[c] = d_temp_acc;
                }

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