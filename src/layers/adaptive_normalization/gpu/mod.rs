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
            // Логит для BatchNorm фиксирован и равен 0

            // Вычисляем статистики по строкам (для LayerNorm и RMSNorm)
            let mut row_mean = vec![0.0f32; rows];
            let mut row_var = vec![0.0f32; rows];
            let mut row_rms_sq = vec![0.0f32; rows];

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
                let rms_sq = sum_sq / cols as f32;
                row_mean[r] = mean;
                row_var[r] = var.max(0.0f32); // защита от отрицательной дисперсии
                row_rms_sq[r] = rms_sq;
            }

            // Вычисляем статистики по столбцам (для BatchNorm)
            let mut col_mean = vec![0.0f32; cols];
            let mut col_var = vec![0.0f32; cols];
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

            // Для каждого признака (столбца) вычисляем веса softmax и выход
            for c in 0..cols {
                let logit_ln = p[logits_ln_start + c];
                let logit_rms = p[logits_rms_start + c];
                let logit_bn = 0.0f32; // фиксированный логит для BN

                // Устойчивый softmax
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

                    // LayerNorm
                    let ln = (x_val - row_mean[r]) / (row_var[r] + eps).sqrt() * gamma_ln + beta_ln;
                    // RMSNorm
                    let rms = x_val / (row_rms_sq[r] + eps).sqrt() * gamma_rms;
                    // BatchNorm
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

                // Смещения
                let ln_gamma_start = base;
                let ln_beta_start = ln_gamma_start + f;
                let rms_gamma_start = ln_beta_start + f;
                let bn_gamma_start = rms_gamma_start + f;
                let bn_beta_start = bn_gamma_start + f;
                let logits_ln_start = bn_beta_start + f;
                let logits_rms_start = logits_ln_start + f;

                // Статистики (такие же, как в forward)
                let mut row_mean = vec![0.0f32; rows];
                let mut row_var = vec![0.0f32; rows];
                let mut row_rms_sq = vec![0.0f32; rows];
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

                let mut col_mean = vec![0.0f32; cols];
                let mut col_var = vec![0.0f32; cols];
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

                // Инициализируем градиенты параметров нулями
                for i in 0..(7 * f) {
                    gp[base + i] = 0.0f32;
                }

                // Обнуляем gi
                for i in 0..(rows * cols) {
                    gi[i] = 0.0f32;
                }

                // Вспомогательные массивы для хранения весов и параметров каждого признака,
                // чтобы не пересчитывать многократно
                let mut w_ln = vec![0.0f32; cols];
                let mut w_rms = vec![0.0f32; cols];
                let mut w_bn = vec![0.0f32; cols];
                let mut gamma_ln = vec![0.0f32; cols];
                let mut beta_ln = vec![0.0f32; cols];
                let mut gamma_rms = vec![0.0f32; cols];
                let mut gamma_bn = vec![0.0f32; cols];
                let mut beta_bn = vec![0.0f32; cols];

                // Вычисляем веса softmax для каждого признака
                for c in 0..cols {
                    let logit_ln = p[logits_ln_start + c];
                    let logit_rms = p[logits_rms_start + c];
                    let logit_bn = 0.0f32;
                    let max_logit = logit_ln.max(logit_rms).max(logit_bn);
                    let exp_ln = (logit_ln - max_logit).exp();
                    let exp_rms = (logit_rms - max_logit).exp();
                    let exp_bn = (logit_bn - max_logit).exp();
                    let sum_exp = exp_ln + exp_rms + exp_bn;
                    w_ln[c] = exp_ln / sum_exp;
                    w_rms[c] = exp_rms / sum_exp;
                    w_bn[c] = exp_bn / sum_exp;

                    gamma_ln[c] = p[ln_gamma_start + c];
                    beta_ln[c] = p[ln_beta_start + c];
                    gamma_rms[c] = p[rms_gamma_start + c];
                    gamma_bn[c] = p[bn_gamma_start + c];
                    beta_bn[c] = p[bn_beta_start + c];
                }

                // Локальные накопители для градиентов по параметрам нормализации
                let mut d_gamma_ln = vec![0.0f32; cols];
                let mut d_beta_ln = vec![0.0f32; cols];
                let mut d_gamma_rms = vec![0.0f32; cols];
                let mut d_gamma_bn = vec![0.0f32; cols];
                let mut d_beta_bn = vec![0.0f32; cols];

                // Первый проход: вычисляем выходы ветвей и накапливаем градиенты
                // по параметрам нормализации, а также прямые производные по входу.
                // Сохраняем выходы ветвей в отдельные векторы для использования позже.
                let mut ln_vals = vec![0.0f32; rows * cols];
                let mut rms_vals = vec![0.0f32; rows * cols];
                let mut bn_vals = vec![0.0f32; rows * cols];

                for c in 0..cols {
                    for r in 0..rows {
                        let idx = c * rows + r;
                        let x_val = x[idx];
                        let mean_r = row_mean[r];
                        let var_r = row_var[r];
                        let rms_r = row_rms_sq[r];
                        let mean_c = col_mean[c];
                        let var_c = col_var[c];

                        let ln = (x_val - mean_r) / (var_r + eps).sqrt() * gamma_ln[c] + beta_ln[c];
                        let rms = x_val / (rms_r + eps).sqrt() * gamma_rms[c];
                        let bn = (x_val - mean_c) / (var_c + eps).sqrt() * gamma_bn[c] + beta_bn[c];

                        ln_vals[idx] = ln;
                        rms_vals[idx] = rms;
                        bn_vals[idx] = bn;

                        let gout = go[idx];

                        // Градиенты по параметрам нормализаций
                        d_gamma_ln[c] += gout * w_ln[c] * (x_val - mean_r) / (var_r + eps).sqrt();
                        d_beta_ln[c] += gout * w_ln[c];
                        d_gamma_rms[c] += gout * w_rms[c] * x_val / (rms_r + eps).sqrt();
                        d_gamma_bn[c] += gout * w_bn[c] * (x_val - mean_c) / (var_c + eps).sqrt();
                        d_beta_bn[c] += gout * w_bn[c];

                        // Прямой вклад в gi
                        let d_ln_dx = gamma_ln[c] / (var_r + eps).sqrt();
                        let d_rms_dx = gamma_rms[c] / (rms_r + eps).sqrt();
                        let d_bn_dx = gamma_bn[c] / (var_c + eps).sqrt();
                        gi[idx] += gout * (w_ln[c] * d_ln_dx + w_rms[c] * d_rms_dx + w_bn[c] * d_bn_dx);
                    }
                }

                // Второй проход: добавляем вклады от изменения статистик.
                // Сначала вклады по строкам для LayerNorm и RMSNorm.
                for r in 0..rows {
                    let N = cols as f32;
                    let mean_r = row_mean[r];
                    let var_r = row_var[r] + eps;
                    let rms_r = row_rms_sq[r] + eps;
                    let inv_N = 1.0 / N;
                    let sqrt_var = var_r.sqrt();
                    let sqrt_rms = rms_r.sqrt();

                    // Суммы, необходимые для градиентов по статистикам
                    let mut sum_g_ln = 0.0f32;          // Σ_c gout_{c,r} * w_ln[c]
                    let mut sum_g_ln_xhat = 0.0f32;     // Σ_c gout_{c,r} * w_ln[c] * xhat_c
                    let mut sum_g_rms_x = 0.0f32;       // Σ_c gout_{c,r} * w_rms[c] * x_c
                    let mut sum_g_rms_x2 = 0.0f32;      // Σ_c gout_{c,r} * w_rms[c] * x_c^2

                    for c in 0..cols {
                        let idx = c * rows + r;
                        let x_val = x[idx];
                        let gout = go[idx];
                        let xhat = (x_val - mean_r) / sqrt_var;
                        sum_g_ln += gout * w_ln[c];
                        sum_g_ln_xhat += gout * w_ln[c] * xhat;
                        sum_g_rms_x += gout * w_rms[c] * x_val;
                        sum_g_rms_x2 += gout * w_rms[c] * x_val * x_val;
                    }

                    // Для каждого признака c в этой строке добавляем вклад в gi
                    for c in 0..cols {
                        let idx = c * rows + r;
                        let x_val = x[idx];
                        let gout = go[idx];

                        // Вклад LayerNorm через изменение mean и var
                        // d(ln_k)/d(mean) = -gamma_ln_k / sqrt_var
                        // d(mean)/d(x_c) = 1/N
                        // => вклад в gi[c] от mean: -(1/N) * Σ_k gout_k * w_ln_k * gamma_ln_k / sqrt_var
                        // Но gamma_ln_k зависит от k, поэтому нужно суммировать по k.
                        // Здесь мы можем вычислить общую сумму для всей строки и прибавить ко всем элементам строки.
                        // Для точности нужно учесть gamma_ln_k.
                        // Мы уже имеем sum_g_ln, но это без учёта gamma_ln_k.
                        // В общем случае не упрощается, поэтому вычислим дополнительно суммы с gamma.
                        // Чтобы не усложнять, выполним дополнительный цикл по k для каждой строки.
                    }
                }

                // Более точный, но менее оптимальный способ: для каждого элемента (c,r)
                // перебрать все k (признаки) и все r' (примеры), чтобы добавить вклады.
                // Учитывая, что размеры features и batch обычно не слишком велики, это приемлемо.
                // Чтобы сохранить код читаемым, выполним отдельные циклы:
                
                // Вклады LayerNorm и RMSNorm от статистик строк
                for r in 0..rows {
                    for c in 0..cols {
                        let idx = c * rows + r;
                        // Добавляем вклад от изменения row_mean, row_var (LN) и row_rms_sq (RMS)
                        // для всех k (признаков) в этой же строке r
                        let mut contrib = 0.0f32;
                        let N = cols as f32;
                        let var_r = row_var[r] + eps;
                        let rms_r = row_rms_sq[r] + eps;
                        let sqrt_var = var_r.sqrt();
                        let sqrt_rms = rms_r.sqrt();
                        let inv_N = 1.0 / N;

                        for k in 0..cols {
                            let kidx = k * rows + r;
                            let gout_k = go[kidx];
                            let w_ln_k = w_ln[k];
                            let w_rms_k = w_rms[k];
                            let gamma_ln_k = gamma_ln[k];
                            let gamma_rms_k = gamma_rms[k];

                            // x_k в этой строке
                            let x_k = x[kidx];

                            // Производные d(ln_k)/d(mean_r) и d(ln_k)/d(var_r)
                            let dln_dmean = -gamma_ln_k / sqrt_var;
                            let dln_dvar = -0.5 * gamma_ln_k * (x_k - row_mean[r]) / (var_r * sqrt_var);

                            // Производные d(mean_r)/d(x_c) и d(var_r)/d(x_c)
                            let dmean_dx = inv_N;
                            let dvar_dx = 2.0 * (x[idx] - row_mean[r]) * inv_N;

                            contrib += gout_k * w_ln_k * (dln_dmean * dmean_dx + dln_dvar * dvar_dx);

                            // Производная d(rms_k)/d(rms_sq_r)
                            let drms_drmssq = -0.5 * gamma_rms_k * x_k / (rms_r * sqrt_rms);
                            // Производная d(rms_sq_r)/d(x_c)
                            let drmssq_dx = 2.0 * x[idx] * inv_N;
                            contrib += gout_k * w_rms_k * drms_drmssq * drmssq_dx;
                        }

                        gi[idx] += contrib;
                    }
                }

                // Вклады BatchNorm от статистик столбца
                for c in 0..cols {
                    let M = rows as f32;
                    let var_c = col_var[c] + eps;
                    let sqrt_var = var_c.sqrt();
                    let inv_M = 1.0 / M;
                    let mean_c = col_mean[c];

                    // Для каждого r (примера) в этом столбце
                    for r in 0..rows {
                        let idx = c * rows + r;
                        let mut contrib = 0.0f32;
                        for rp in 0..rows {
                            let pidx = c * rows + rp;
                            let gout_p = go[pidx];
                            let w_bn_c = w_bn[c];
                            let gamma_bn_c = gamma_bn[c];

                            let x_p = x[pidx];

                            let dbn_dmean = -gamma_bn_c / sqrt_var;
                            let dbn_dvar = -0.5 * gamma_bn_c * (x_p - mean_c) / (var_c * sqrt_var);

                            let dmean_dx = inv_M;
                            let dvar_dx = 2.0 * (x[idx] - mean_c) * inv_M;

                            contrib += gout_p * w_bn_c * (dbn_dmean * dmean_dx + dbn_dvar * dvar_dx);
                        }
                        gi[idx] += contrib;
                    }
                }

                // Записываем градиенты параметров нормализации
                for c in 0..cols {
                    gp[ln_gamma_start + c] += d_gamma_ln[c];
                    gp[ln_beta_start + c] += d_beta_ln[c];
                    gp[rms_gamma_start + c] += d_gamma_rms[c];
                    gp[bn_gamma_start + c] += d_gamma_bn[c];
                    gp[bn_beta_start + c] += d_beta_bn[c];
                }

                // Градиенты по логитам (только для ln и rms, bn фиксирован)
                // Вычисляем dL/dlogit_ln и dL/dlogit_rms
                for c in 0..cols {
                    let logit_ln = p[logits_ln_start + c];
                    let logit_rms = p[logits_rms_start + c];
                    let logit_bn = 0.0f32;
                    let max_logit = logit_ln.max(logit_rms).max(logit_bn);
                    let exp_ln = (logit_ln - max_logit).exp();
                    let exp_rms = (logit_rms - max_logit).exp();
                    let exp_bn = (logit_bn - max_logit).exp();
                    let sum_exp = exp_ln + exp_rms + exp_bn;
                    let w_ln_c = exp_ln / sum_exp;
                    let w_rms_c = exp_rms / sum_exp;
                    let w_bn_c = exp_bn / sum_exp;

                    let mut dL_dw_ln = 0.0f32;
                    let mut dL_dw_rms = 0.0f32;
                    let mut dL_dw_bn = 0.0f32;
                    for r in 0..rows {
                        let idx = c * rows + r;
                        dL_dw_ln += go[idx] * ln_vals[idx];
                        dL_dw_rms += go[idx] * rms_vals[idx];
                        dL_dw_bn += go[idx] * bn_vals[idx];
                    }

                    // Производные softmax
                    let dw_ln_dlogit_ln = w_ln_c * (1.0 - w_ln_c);
                    let dw_rms_dlogit_ln = -w_rms_c * w_ln_c;
                    let dw_bn_dlogit_ln = -w_bn_c * w_ln_c;
                    let dL_dlogit_ln = dL_dw_ln * dw_ln_dlogit_ln
                                     + dL_dw_rms * dw_rms_dlogit_ln
                                     + dL_dw_bn * dw_bn_dlogit_ln;

                    let dw_ln_dlogit_rms = -w_ln_c * w_rms_c;
                    let dw_rms_dlogit_rms = w_rms_c * (1.0 - w_rms_c);
                    let dw_bn_dlogit_rms = -w_bn_c * w_rms_c;
                    let dL_dlogit_rms = dL_dw_ln * dw_ln_dlogit_rms
                                      + dL_dw_rms * dw_rms_dlogit_rms
                                      + dL_dw_bn * dw_bn_dlogit_rms;

                    gp[logits_ln_start + c] += dL_dlogit_ln;
                    gp[logits_rms_start + c] += dL_dlogit_rms;
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