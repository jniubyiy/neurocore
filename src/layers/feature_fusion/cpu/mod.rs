// src/layers/feature_fusion/cpu/mod.rs

use crate::compute_manager::graph::types::DynamicContext;
use crate::compute_manager::matrix_buffer::MatrixBufferHandle;
use crate::layers::buffered_context::BufferedContext;
use crate::layers::UniversalLayerBuffered;
use crate::model_plan::param_store::ParamSlice;

use super::super::feature_fusion::FeatureFusion;

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
        debug_assert!(slice.start + self.param_len() <= params.rows() * params.cols());

        let ids = [input.id(), output.id(), params.id()];
        input.memory().write().unwrap().with_cpu_slices_mut(&ids, |slices| {
            let (first, rest) = slices.split_at_mut(1);
            let x: &[f32] = &*first[0];
            let (second, rest) = rest.split_at_mut(1);
            let y: &mut [f32] = &mut *second[0];
            let p: &[f32] = &*rest[0];

            let base = slice.start;
            let fin = self.in_features;
            let fout = self.out_features;

            for j in 0..fout {
                // softmax логитов для выхода j
                let mut max_l = f32::NEG_INFINITY;
                for i in 0..fin {
                    let l = p[base + j * fin + i];
                    if l > max_l { max_l = l; }
                }
                let mut sum_exp = 0.0;
                let mut weights = vec![0.0f32; fin];
                for i in 0..fin {
                    let e = (p[base + j * fin + i] - max_l).exp();
                    weights[i] = e;
                    sum_exp += e;
                }
                let inv_sum = 1.0 / sum_exp;
                for i in 0..fin {
                    weights[i] *= inv_sum;
                }

                let bias = p[base + fout * fin + j];

                for r in 0..batch {
                    let mut acc = bias;
                    for i in 0..fin {
                        acc += weights[i] * x[i * batch + r];
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
                let bias_start = base + fout * fin;

                // Инициализируем градиенты параметров нулями
                let mut grad_logits = vec![0.0f32; fout * fin];
                let mut grad_bias = vec![0.0f32; fout];

                // Градиент по входу накапливаем
                for i in 0..fin {
                    for r in 0..batch {
                        gi[i * batch + r] = 0.0;
                    }
                }

                for j in 0..fout {
                    // Повторяем softmax
                    let mut max_l = f32::NEG_INFINITY;
                    for i in 0..fin {
                        let l = p[logits_start + j * fin + i];
                        if l > max_l { max_l = l; }
                    }
                    let mut sum_exp = 0.0;
                    let mut weights = vec![0.0f32; fin];
                    for i in 0..fin {
                        let e = (p[logits_start + j * fin + i] - max_l).exp();
                        weights[i] = e;
                        sum_exp += e;
                    }
                    let inv_sum = 1.0 / sum_exp;
                    for i in 0..fin { weights[i] *= inv_sum; }

                    for r in 0..batch {
                        let gout = go[j * batch + r];
                        grad_bias[j] += gout;

                        for i in 0..fin {
                            let x_val = x[i * batch + r];
                            // Градиент по входу
                            gi[i * batch + r] += gout * weights[i];
                            // Градиент по логитам (для softmax)
                            // dL/dlogit_j_i = sum_r gout_r * x_i * (w_i * (1 - w_i)) - для cross-entropy?
                            // Точная формула: dL/dlogit_i = w_i * (x_i - y_j) * sum_r gout_r
                            // Но так как y_j включает bias, и bias тоже обучаемый, то
                            // dL/dlogit_i = w_i * (x_i * sum_r gout_r - sum_r gout_r * y_j?) 
                            // Упрощённо используем: dL/dlogit_i = sum_r gout_r * x_i * w_i * (1 - w_i)
                            // Это правильно только если выход j единственный и без смешения с другими выходами.
                            // В нашем случае выходы независимы, поэтому можно считать для каждого j отдельно.
                            // Вклад в градиент логита i от выхода j:
                            // dL/dlogit_{j,i} = sum_r gout_{j,r} * x_{i,r} * w_{j,i} * (1 - w_{j,i})
                            grad_logits[j * fin + i] += gout * x_val * weights[i] * (1.0 - weights[i]);
                        }
                    }
                }

                // Записываем градиенты
                for j in 0..fout {
                    gp[bias_start + j] = grad_bias[j];
                    for i in 0..fin {
                        gp[logits_start + j * fin + i] = grad_logits[j * fin + i];
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