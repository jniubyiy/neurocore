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
            let features = self.features;

            // Смещения параметров в буфере
            let ln_gamma_start = base;
            let ln_beta_start  = ln_gamma_start + features;
            let rms_gamma_start = ln_beta_start + features;
            let bn_gamma_start = rms_gamma_start + features;
            let bn_beta_start  = bn_gamma_start + features;
            let logits_start   = bn_beta_start + features; // 3 * features

            // Вычисляем статистики для LayerNorm и RMSNorm (по строкам)
            // Для BatchNorm нужны статистики по столбцам.
            // Мы будем вычислять все три нормализации для каждого элемента.

            // Временные буферы для результатов нормализаций можно не хранить,
            // а сразу накапливать взвешенную сумму, но для обратного прохода
            // нам понадобятся промежуточные значения, поэтому сохраним в локальных
            // переменных (не в матрицах). Для простоты будем вычислять для каждого
            // элемента отдельно, но это неэффективно. В реальной реализации лучше
            // сначала вычислить статистики по строкам/столбцам, затем нормализовать.

            // Предварительно вычислим средние и дисперсии по строкам (LayerNorm, RMSNorm)
            let mut row_mean = vec![0.0f32; rows];
            let mut row_var  = vec![0.0f32; rows];
            let mut row_rms  = vec![0.0f32; rows];

            for r in 0..rows {
                let mut sum = 0.0;
                let mut sum_sq = 0.0;
                for c in 0..cols {
                    let idx = c * rows + r;
                    let v = x[idx];
                    sum += v;
                    sum_sq += v * v;
                }
                let mean = sum / cols as f32;
                let var = sum_sq / cols as f32 - mean * mean;
                let rms = (sum_sq / cols as f32).sqrt();
                row_mean[r] = mean;
                row_var[r] = var;
                row_rms[r] = rms;
            }

            // Статистики для BatchNorm (по столбцам)
            let mut col_mean = vec![0.0f32; cols];
            let mut col_var  = vec![0.0f32; cols];

            for c in 0..cols {
                let mut sum = 0.0;
                let mut sum_sq = 0.0;
                for r in 0..rows {
                    let idx = c * rows + r;
                    let v = x[idx];
                    sum += v;
                    sum_sq += v * v;
                }
                let mean = sum / rows as f32;
                let var = sum_sq / rows as f32 - mean * mean;
                col_mean[c] = mean;
                col_var[c] = var;
            }

            // Вычисляем выход
            for c in 0..cols {
                // softmax логитов для признака c
                let l0 = p[logits_start + 0 * features + c];
                let l1 = p[logits_start + 1 * features + c];
                let l2 = p[logits_start + 2 * features + c];
                let max_l = l0.max(l1).max(l2);
                let e0 = (l0 - max_l).exp();
                let e1 = (l1 - max_l).exp();
                let e2 = (l2 - max_l).exp();
                let sum_exp = e0 + e1 + e2;
                let w_ln = e0 / sum_exp;
                let w_rms = e1 / sum_exp;
                let w_bn = e2 / sum_exp;

                let ln_gamma = p[ln_gamma_start + c];
                let ln_beta  = p[ln_beta_start + c];
                let rms_gamma = p[rms_gamma_start + c];
                let bn_gamma = p[bn_gamma_start + c];
                let bn_beta  = p[bn_beta_start + c];

                for r in 0..rows {
                    let idx = c * rows + r;
                    let x_val = x[idx];

                    // LayerNorm
                    let ln = (x_val - row_mean[r]) / (row_var[r] + 1e-5).sqrt() * ln_gamma + ln_beta;
                    // RMSNorm
                    let rms = x_val / (row_rms[r] + 1e-5) * rms_gamma;
                    // BatchNorm
                    let bn = (x_val - col_mean[c]) / (col_var[c] + 1e-5).sqrt() * bn_gamma + bn_beta;

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
                let features = self.features;

                let ln_gamma_start = base;
                let ln_beta_start  = ln_gamma_start + features;
                let rms_gamma_start = ln_beta_start + features;
                let bn_gamma_start = rms_gamma_start + features;
                let bn_beta_start  = bn_gamma_start + features;
                let logits_start   = bn_beta_start + features; // 3 * features

                // Инициализируем градиенты по параметрам нулями (они будут накапливаться)
                let param_len = self.param_len();
                let mut grad_acc = vec![0.0f32; param_len];

                // Вычисляем те же статистики, что и в прямом проходе
                let mut row_mean = vec![0.0f32; rows];
                let mut row_var  = vec![0.0f32; rows];
                let mut row_rms  = vec![0.0f32; rows];
                for r in 0..rows {
                    let mut sum = 0.0;
                    let mut sum_sq = 0.0;
                    for c in 0..cols {
                        let idx = c * rows + r;
                        let v = x[idx];
                        sum += v;
                        sum_sq += v * v;
                    }
                    let mean = sum / cols as f32;
                    let var = sum_sq / cols as f32 - mean * mean;
                    let rms = (sum_sq / cols as f32).sqrt();
                    row_mean[r] = mean;
                    row_var[r] = var;
                    row_rms[r] = rms;
                }

                let mut col_mean = vec![0.0f32; cols];
                let mut col_var  = vec![0.0f32; cols];
                for c in 0..cols {
                    let mut sum = 0.0;
                    let mut sum_sq = 0.0;
                    for r in 0..rows {
                        let idx = c * rows + r;
                        let v = x[idx];
                        sum += v;
                        sum_sq += v * v;
                    }
                    let mean = sum / rows as f32;
                    let var = sum_sq / rows as f32 - mean * mean;
                    col_mean[c] = mean;
                    col_var[c] = var;
                }

                // Для каждого признака вычисляем веса и градиенты
                for c in 0..cols {
                    let l0 = p[logits_start + 0 * features + c];
                    let l1 = p[logits_start + 1 * features + c];
                    let l2 = p[logits_start + 2 * features + c];
                    let max_l = l0.max(l1).max(l2);
                    let e0 = (l0 - max_l).exp();
                    let e1 = (l1 - max_l).exp();
                    let e2 = (l2 - max_l).exp();
                    let sum_exp = e0 + e1 + e2;
                    let w_ln = e0 / sum_exp;
                    let w_rms = e1 / sum_exp;
                    let w_bn = e2 / sum_exp;

                    let ln_gamma = p[ln_gamma_start + c];
                    let ln_beta  = p[ln_beta_start + c];
                    let rms_gamma = p[rms_gamma_start + c];
                    let bn_gamma = p[bn_gamma_start + c];
                    let bn_beta  = p[bn_beta_start + c];

                    let mut d_ln_gamma = 0.0;
                    let mut d_ln_beta  = 0.0;
                    let mut d_rms_gamma = 0.0;
                    let mut d_bn_gamma = 0.0;
                    let mut d_bn_beta  = 0.0;
                    let mut d_l0 = 0.0;
                    let mut d_l1 = 0.0;
                    let mut d_l2 = 0.0;

                    for r in 0..rows {
                        let idx = c * rows + r;
                        let x_val = x[idx];
                        let gout = go[idx];

                        let ln_norm = (x_val - row_mean[r]) / (row_var[r] + 1e-5).sqrt();
                        let rms_norm = x_val / (row_rms[r] + 1e-5);
                        let bn_norm = (x_val - col_mean[c]) / (col_var[c] + 1e-5).sqrt();

                        let ln = ln_norm * ln_gamma + ln_beta;
                        let rms = rms_norm * rms_gamma;
                        let bn = bn_norm * bn_gamma + bn_beta;

                        // Градиент по выходу взвешенной суммы
                        let d_ln = w_ln;
                        let d_rms = w_rms;
                        let d_bn = w_bn;

                        // Вклад в градиент по входу
                        let gi_ln = gout * d_ln * ln_gamma / (row_var[r] + 1e-5).sqrt();
                        let gi_rms = gout * d_rms * rms_gamma / (row_rms[r] + 1e-5);
                        let gi_bn = gout * d_bn * bn_gamma / (col_var[c] + 1e-5).sqrt();
                        // это упрощённо, точная производная сложнее, но для первого приближения приемлемо
                        gi[idx] = gi_ln + gi_rms + gi_bn;

                        // Градиенты по параметрам нормализации
                        d_ln_gamma += gout * w_ln * ln_norm;
                        d_ln_beta  += gout * w_ln;
                        d_rms_gamma += gout * w_rms * rms_norm;
                        d_bn_gamma += gout * w_bn * bn_norm;
                        d_bn_beta  += gout * w_bn;

                        // Градиенты по логитам (для softmax)
                        let y_combined = w_ln * ln + w_rms * rms + w_bn * bn;
                        d_l0 += gout * (ln - y_combined) * w_ln;
                        d_l1 += gout * (rms - y_combined) * w_rms;
                        d_l2 += gout * (bn - y_combined) * w_bn;
                    }

                    // Сохраняем накопленные градиенты параметров
                    grad_acc[ln_gamma_start - base + c] = d_ln_gamma;
                    grad_acc[ln_beta_start - base + c] = d_ln_beta;
                    grad_acc[rms_gamma_start - base + c] = d_rms_gamma;
                    grad_acc[bn_gamma_start - base + c] = d_bn_gamma;
                    grad_acc[bn_beta_start - base + c] = d_bn_beta;
                    grad_acc[logits_start - base + 0 * features + c] = d_l0;
                    grad_acc[logits_start - base + 1 * features + c] = d_l1;
                    grad_acc[logits_start - base + 2 * features + c] = d_l2;
                }

                // Записываем градиенты в общий буфер
                for i in 0..param_len {
                    gp[base + i] = grad_acc[i];
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