// src/layers/feature_fusion/cpu/mod.rs

use crate::compute_manager::graph::types::DynamicContext;
use crate::compute_manager::matrix_buffer::MatrixBufferHandle;
use crate::layers::buffered_context::BufferedContext;
use crate::layers::UniversalLayerBuffered;
use crate::model_plan::param_store::ParamSlice;

use super::super::feature_fusion::FeatureFusion;

/// Минимальное значение T_eff. Гарантирует, что 1/T конечно.
const TEMP_EPS: f32 = 1e-6;

/// Численно устойчивый softmax от `L / T` для одного выхода слоя.
///
/// Возвращает вектор весов длины `fin` в буфер `w` (перезаписывает его).
///
/// Аргументы:
/// * `p`   — весь буфер параметров (родитель).
/// * `logits_offset` — начало логитов выхода j в `p`.
/// * `fin` — число входных признаков.
/// * `t_eff` — эффективная температура (положительная).
/// * `w` — выходной буфер весов, длина `fin`.
#[inline]
fn softmax_over_logits(
    p: &[f32],
    logits_offset: usize,
    fin: usize,
    t_eff: f32,
    w: &mut [f32],
) {
    debug_assert_eq!(w.len(), fin);
    let inv_t = 1.0 / t_eff;

    // Численно устойчивый softmax: сначала max(L), потом exp((L-max)/T).
    // Сдвиг на max(L) не меняет softmax, поскольку сдвиг одинаков для всех i.
    let mut max_l = f32::NEG_INFINITY;
    for i in 0..fin {
        let l = p[logits_offset + i];
        if l > max_l {
            max_l = l;
        }
    }

    let mut sum_exp = 0.0f32;
    for i in 0..fin {
        let e = ((p[logits_offset + i] - max_l) * inv_t).exp();
        w[i] = e;
        sum_exp += e;
    }

    let inv_sum = 1.0 / sum_exp;
    for wi in w.iter_mut() {
        *wi *= inv_sum;
    }
}

impl UniversalLayerBuffered for FeatureFusion {
    fn forward_buffered(
        &self,
        input: &MatrixBufferHandle,
        output: &MatrixBufferHandle,
        params: &MatrixBufferHandle,
        slice: &ParamSlice,
    ) {
        let batch = input.rows();
        let cols_in = input.cols();
        let cols_out = self.out_features;
        debug_assert_eq!(cols_in, self.in_features);
        debug_assert_eq!(output.rows(), batch);
        debug_assert_eq!(output.cols(), cols_out);
        debug_assert!(slice.start + self.param_len() <= params.rows() * params.cols());

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
                let fin = self.in_features;
                let fout = self.out_features;

                let logits_start = base;
                let temp_start = base + fout * fin;

                let mut w = vec![0.0f32; fin];

                for j in 0..fout {
                    let logits_offset = logits_start + j * fin;
                    let t_raw = p[temp_start + j];
                    let t_eff = t_raw.abs() + TEMP_EPS;

                    softmax_over_logits(p, logits_offset, fin, t_eff, &mut w);

                    // y_{j,r} = Σ_i w_i · x_{i,r} — чистая convex combination,
                    // без bias.
                    for r in 0..batch {
                        let mut acc = 0.0f32;
                        for i in 0..fin {
                            acc += w[i] * x[i * batch + r];
                        }
                        y[j * batch + r] = acc;
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
            BufferedContext::FeatureFusion { input } => input,
            _ => panic!("Expected FeatureFusion context"),
        };

        let batch = grad_output.rows();
        let fin = self.in_features;
        let fout = self.out_features;
        debug_assert_eq!(grad_output.cols(), fout);
        debug_assert_eq!(input_handle.cols(), fin);
        debug_assert_eq!(grad_input.rows(), batch);
        debug_assert_eq!(grad_input.cols(), fin);
        debug_assert!(slice.start + self.param_len() <= params.rows() * params.cols());
        debug_assert!(
            slice.start + self.param_len() <= grad_params.rows() * grad_params.cols()
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
                let logits_start = base;
                let temp_start = base + fout * fin;

                // Обнуляем накопители градиентов параметров и входной градиент.
                for i in 0..self.param_len() {
                    gp[base + i] = 0.0;
                }
                for v in gi.iter_mut() {
                    *v = 0.0;
                }

                let mut w = vec![0.0f32; fin];
                let mut u_local = vec![0.0f32; batch];
                let mut dz = vec![0.0f32; fin];

                for j in 0..fout {
                    let logits_offset = logits_start + j * fin;
                    let t_raw = p[temp_start + j];
                    let t_eff = t_raw.abs() + TEMP_EPS;
                    let inv_t = 1.0 / t_eff;

                    // 1. Пересчёт softmax для выхода j — идентичен forward.
                    softmax_over_logits(p, logits_offset, fin, t_eff, &mut w);

                    // 2. u_{j,r} = Σ_i w_i · x_{i,r} — attention output.
                    for r in 0..batch {
                        let mut acc = 0.0f32;
                        for i in 0..fin {
                            acc += w[i] * x[i * batch + r];
                        }
                        u_local[r] = acc;
                    }

                    // 3. dz_k = ∂L/∂z_{j,k}, где z = L / T_eff.
                    for dzk in dz.iter_mut() {
                        *dzk = 0.0;
                    }
                    for r in 0..batch {
                        let gout = go[j * batch + r];
                        let u_r = u_local[r];
                        for i in 0..fin {
                            let x_val = x[i * batch + r];
                            dz[i] += gout * w[i] * (x_val - u_r);
                        }
                    }

                    // 4. Градиент по логитам: ∂L/∂L_k = dz_k / T_eff.
                    //    Градиент по температуре:
                    //      ∂L/∂T = −(1/T²) · Σ_k L_k · dz_k,
                    //    и через |·| для T_raw:
                    //      ∂L/∂T_raw = ∂L/∂T · sign(T_raw).
                    let mut dot_ldz = 0.0f32;
                    for i in 0..fin {
                        let l_i = p[logits_offset + i];
                        dot_ldz += l_i * dz[i];
                        gp[logits_start + j * fin + i] = dz[i] * inv_t;
                    }
                    let sign_t = if t_raw > 0.0 {
                        1.0
                    } else if t_raw < 0.0 {
                        -1.0
                    } else {
                        0.0
                    };
                    gp[temp_start + j] = -sign_t * dot_ldz * inv_t * inv_t;

                    // 5. Градиент по входу: ∂L/∂x_{i,r} += go_{j,r} · w_i.
                    //    Не зависит от T, поскольку w — функция от x через
                    //    softmax по логитам, а сами логиты от x не зависят.
                    for r in 0..batch {
                        let gout = go[j * batch + r];
                        for i in 0..fin {
                            gi[i * batch + r] += gout * w[i];
                        }
                    }
                }
            });
    }

    fn param_len(&self) -> usize {
        self.out_features * (self.in_features + 1)
    }

    fn input_features(&self) -> usize {
        self.in_features
    }

    fn output_features(&self) -> usize {
        self.out_features
    }
}