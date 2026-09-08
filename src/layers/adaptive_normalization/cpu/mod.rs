// src/layers/adaptive_normalization/cpu/mod.rs

use crate::compute_manager::graph::types::DynamicContext;
use crate::compute_manager::matrix_buffer::MatrixBufferHandle;
use crate::layers::buffered_context::BufferedContext;
use crate::layers::UniversalLayerBuffered;
use crate::model_plan::param_store::ParamSlice;

use super::super::adaptive_normalization::AdaptiveNormalization;

impl UniversalLayerBuffered for AdaptiveNormalization {
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
            "AdaptiveNormalization: parameter slice out of bounds"
        );

        let ids = [input.id(), output.id(), params.id()];
        input.memory().write().unwrap().with_cpu_slices_mut(&ids, |slices| {
            let (first, rest) = slices.split_at_mut(1);
            let x: &[f32] = &*first[0];
            let (second, rest) = rest.split_at_mut(1);
            let y: &mut [f32] = &mut *second[0];
            let p: &[f32] = &*rest[0];

            let base = slice.start;
            let f = self.features;
            let eps = 1e-5f32;

            // Смещения параметров (всего 7f элементов)
            let ln_gamma_start = base;
            let ln_beta_start = ln_gamma_start + f;
            let rms_gamma_start = ln_beta_start + f;
            let bn_gamma_start = rms_gamma_start + f;
            let bn_beta_start = bn_gamma_start + f;
            let logits_ln_start = bn_beta_start + f;
            let logits_rms_start = logits_ln_start + f;

            // Статистики по строкам и столбцам
            let mut row_mean = vec![0.0f32; rows];
            let mut row_var = vec![0.0f32; rows];
            let mut row_rms_sq = vec![0.0f32; rows];
            let mut col_mean = vec![0.0f32; cols];
            let mut col_var = vec![0.0f32; cols];

            for r in 0..rows {
                let mut sum = 0.0f32;
                let mut sum_sq = 0.0f32;
                for c in 0..cols {
                    let idx = c * rows + r;
                    let v = x[idx];
                    sum += v;
                    sum_sq += v * v;
                }
                let mean = sum / cols as f32;
                let var = sum_sq / cols as f32 - mean * mean;
                row_mean[r] = mean;
                row_var[r] = var.max(0.0f32);
                row_rms_sq[r] = sum_sq / cols as f32;
            }
            for c in 0..cols {
                let mut sum = 0.0f32;
                let mut sum_sq = 0.0f32;
                for r in 0..rows {
                    let idx = c * rows + r;
                    let v = x[idx];
                    sum += v;
                    sum_sq += v * v;
                }
                let mean = sum / rows as f32;
                let var = sum_sq / rows as f32 - mean * mean;
                col_mean[c] = mean;
                col_var[c] = var.max(0.0f32);
            }

            // Основной цикл
            for c in 0..cols {
                let logit_ln = p[logits_ln_start + c];
                let logit_rms = p[logits_rms_start + c];
                let logit_bn = 0.0f32;

                let max_logit = logit_ln.max(logit_rms).max(logit_bn);
                let exp_ln = (logit_ln - max_logit).exp();
                let exp_rms = (logit_rms - max_logit).exp();
                let exp_bn = (logit_bn - max_logit).exp();
                let sum_exp = exp_ln + exp_rms + exp_bn;
                let w_ln = exp_ln / sum_exp;
                let w_rms = exp_rms / sum_exp;
                let w_bn = exp_bn / sum_exp;

                let gamma_ln = p[ln_gamma_start + c];
                let beta_ln = p[ln_beta_start + c];
                let gamma_rms = p[rms_gamma_start + c];
                let gamma_bn = p[bn_gamma_start + c];
                let beta_bn = p[bn_beta_start + c];

                for r in 0..rows {
                    let idx = c * rows + r;
                    let x_val = x[idx];

                    let ln = (x_val - row_mean[r]) / (row_var[r] + eps).sqrt() * gamma_ln + beta_ln;
                    let rms = x_val / (row_rms_sq[r] + eps).sqrt() * gamma_rms;
                    let bn = (x_val - col_mean[c]) / (col_var[c] + eps).sqrt() * gamma_bn + beta_bn;

                    y[idx] = w_ln * ln + w_rms * rms + w_bn * bn;
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
            BufferedContext::AdaptiveNormalization { input } => input,
            _ => panic!("Expected AdaptiveNormalization context"),
        };

        let rows = grad_output.rows();
        let cols = grad_output.cols();
        debug_assert_eq!(cols, self.features);
        debug_assert_eq!(rows, input_handle.rows());
        debug_assert!(
            slice.start + self.param_len() <= params.rows() * params.cols(),
            "AdaptiveNormalization backward: parameter slice out of bounds"
        );
        debug_assert!(
            slice.start + self.param_len() <= grad_params.rows() * grad_params.cols(),
            "AdaptiveNormalization backward: grad parameter slice out of bounds"
        );

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
                let f = self.features;
                let eps = 1e-5f32;

                // Смещения параметров
                let ln_gamma_start = base;
                let ln_beta_start = ln_gamma_start + f;
                let rms_gamma_start = ln_beta_start + f;
                let bn_gamma_start = rms_gamma_start + f;
                let bn_beta_start = bn_gamma_start + f;
                let logits_ln_start = bn_beta_start + f;
                let logits_rms_start = logits_ln_start + f;

                // Инициализируем градиенты параметров нулями
                for i in 0..(7 * f) {
                    gp[base + i] = 0.0f32;
                }

                // Вычисляем статистики (аналогично forward)
                let mut row_mean = vec![0.0f32; rows];
                let mut row_var = vec![0.0f32; rows];
                let mut row_rms_sq = vec![0.0f32; rows];
                let mut col_mean = vec![0.0f32; cols];
                let mut col_var = vec![0.0f32; cols];

                for r in 0..rows {
                    let mut sum = 0.0f32;
                    let mut sum_sq = 0.0f32;
                    for c in 0..cols {
                        let idx = c * rows + r;
                        let v = x[idx];
                        sum += v;
                        sum_sq += v * v;
                    }
                    let mean = sum / cols as f32;
                    let var = sum_sq / cols as f32 - mean * mean;
                    row_mean[r] = mean;
                    row_var[r] = var.max(0.0f32);
                    row_rms_sq[r] = sum_sq / cols as f32;
                }
                for c in 0..cols {
                    let mut sum = 0.0f32;
                    let mut sum_sq = 0.0f32;
                    for r in 0..rows {
                        let idx = c * rows + r;
                        let v = x[idx];
                        sum += v;
                        sum_sq += v * v;
                    }
                    let mean = sum / rows as f32;
                    let var = sum_sq / rows as f32 - mean * mean;
                    col_mean[c] = mean;
                    col_var[c] = var.max(0.0f32);
                }

                // Вычисляем w_ln, w_rms, w_bn и промежуточные градиенты ветвей
                // Также сразу накапливаем градиенты параметров и промежуточные суммы
                let mut grad_gamma_ln = vec![0.0f32; f];
                let mut grad_beta_ln = vec![0.0f32; f];
                let mut grad_gamma_rms = vec![0.0f32; f];
                let mut grad_gamma_bn = vec![0.0f32; f];
                let mut grad_beta_bn = vec![0.0f32; f];
                let mut grad_logits_ln = vec![0.0f32; f];
                let mut grad_logits_rms = vec![0.0f32; f];

                // Промежуточные значения dln, drms, dbn для каждого элемента
                let mut dln = vec![0.0f32; rows * cols];
                let mut drms = vec![0.0f32; rows * cols];
                let mut dbn = vec![0.0f32; rows * cols];

                // Заполняем dln, drms, dbn и накапливаем некоторые суммы
                for c in 0..cols {
                    let logit_ln = p[logits_ln_start + c];
                    let logit_rms = p[logits_rms_start + c];
                    let logit_bn = 0.0f32;

                    let max_logit = logit_ln.max(logit_rms).max(logit_bn);
                    let exp_ln = (logit_ln - max_logit).exp();
                    let exp_rms = (logit_rms - max_logit).exp();
                    let exp_bn = (logit_bn - max_logit).exp();
                    let sum_exp = exp_ln + exp_rms + exp_bn;
                    let w_ln = exp_ln / sum_exp;
                    let w_rms = exp_rms / sum_exp;
                    let w_bn = exp_bn / sum_exp;

                    let gamma_ln = p[ln_gamma_start + c];
                    let beta_ln = p[ln_beta_start + c];
                    let gamma_rms = p[rms_gamma_start + c];
                    let gamma_bn = p[bn_gamma_start + c];
                    let beta_bn = p[bn_beta_start + c];

                    for r in 0..rows {
                        let idx = c * rows + r;
                        let gout = go[idx];
                        let x_val = x[idx];

                        let ln_val = (x_val - row_mean[r]) / (row_var[r] + eps).sqrt() * gamma_ln + beta_ln;
                        let rms_val = x_val / (row_rms_sq[r] + eps).sqrt() * gamma_rms;
                        let bn_val = (x_val - col_mean[c]) / (col_var[c] + eps).sqrt() * gamma_bn + beta_bn;

                        let dln_val = gout * w_ln;
                        let drms_val = gout * w_rms;
                        let dbn_val = gout * w_bn;

                        dln[idx] = dln_val;
                        drms[idx] = drms_val;
                        dbn[idx] = dbn_val;

                        // Накапливаем градиенты параметров
                        grad_gamma_ln[c] += dln_val * (x_val - row_mean[r]) / (row_var[r] + eps).sqrt();
                        grad_beta_ln[c] += dln_val;
                        grad_gamma_rms[c] += drms_val * x_val / (row_rms_sq[r] + eps).sqrt();
                        grad_gamma_bn[c] += dbn_val * (x_val - col_mean[c]) / (col_var[c] + eps).sqrt();
                        grad_beta_bn[c] += dbn_val;

                        // Градиенты по логитам
                        grad_logits_ln[c] += gout * w_ln * (ln_val - (w_ln * ln_val + w_rms * rms_val + w_bn * bn_val));
                        grad_logits_rms[c] += gout * w_rms * (rms_val - (w_ln * ln_val + w_rms * rms_val + w_bn * bn_val));
                    }
                }

                // Вычисляем суммы для корректировок статистик
                let mut sum_dln_per_row = vec![0.0f32; rows];
                let mut sum_dln_x_per_row = vec![0.0f32; rows];
                let mut sum_drms_x_per_row = vec![0.0f32; rows];
                let mut sum_dbn_per_col = vec![0.0f32; cols];
                let mut sum_dbn_x_per_col = vec![0.0f32; cols];

                for r in 0..rows {
                    let mut s1 = 0.0f32;
                    let mut s2 = 0.0f32;
                    let mut s3 = 0.0f32;
                    for c in 0..cols {
                        let idx = c * rows + r;
                        s1 += dln[idx];
                        s2 += dln[idx] * (x[idx] - row_mean[r]);
                        s3 += drms[idx] * x[idx];
                    }
                    sum_dln_per_row[r] = s1;
                    sum_dln_x_per_row[r] = s2;
                    sum_drms_x_per_row[r] = s3;
                }
                for c in 0..cols {
                    let mut s1 = 0.0f32;
                    let mut s2 = 0.0f32;
                    for r in 0..rows {
                        let idx = c * rows + r;
                        s1 += dbn[idx];
                        s2 += dbn[idx] * (x[idx] - col_mean[c]);
                    }
                    sum_dbn_per_col[c] = s1;
                    sum_dbn_x_per_col[c] = s2;
                }

                // Вычисляем градиент по входу gi
                for c in 0..cols {
                    let gamma_ln = p[ln_gamma_start + c];
                    let gamma_rms = p[rms_gamma_start + c];
                    let gamma_bn = p[bn_gamma_start + c];

                    for r in 0..rows {
                        let idx = c * rows + r;
                        let x_val = x[idx];

                        // Вклад от LayerNorm
                        let inv_std_ln = 1.0 / (row_var[r] + eps).sqrt();
                        let term_ln = gamma_ln * inv_std_ln * (
                            dln[idx]
                            - sum_dln_per_row[r] / cols as f32
                            - (x_val - row_mean[r]) / (cols as f32 * (row_var[r] + eps)) * sum_dln_x_per_row[r]
                        );

                        // Вклад от RMSNorm
                        let inv_std_rms = 1.0 / (row_rms_sq[r] + eps).sqrt();
                        let term_rms = gamma_rms * inv_std_rms * (
                            drms[idx]
                            - (x_val / (cols as f32 * (row_rms_sq[r] + eps))) * sum_drms_x_per_row[r]
                        );

                        // Вклад от BatchNorm
                        let inv_std_bn = 1.0 / (col_var[c] + eps).sqrt();
                        let term_bn = gamma_bn * inv_std_bn * (
                            dbn[idx]
                            - sum_dbn_per_col[c] / rows as f32
                            - (x_val - col_mean[c]) / (rows as f32 * (col_var[c] + eps)) * sum_dbn_x_per_col[c]
                        );

                        gi[idx] = term_ln + term_rms + term_bn;
                    }
                }

                // Записываем градиенты параметров
                for c in 0..f {
                    gp[ln_gamma_start + c] = grad_gamma_ln[c];
                    gp[ln_beta_start + c] = grad_beta_ln[c];
                    gp[rms_gamma_start + c] = grad_gamma_rms[c];
                    gp[bn_gamma_start + c] = grad_gamma_bn[c];
                    gp[bn_beta_start + c] = grad_beta_bn[c];
                    gp[logits_ln_start + c] = grad_logits_ln[c];
                    gp[logits_rms_start + c] = grad_logits_rms[c];
                }
            });
    }

    fn param_len(&self) -> usize {
        7 * self.features
    }

    fn input_features(&self) -> usize {
        self.features
    }

    fn output_features(&self) -> usize {
        self.features
    }
}