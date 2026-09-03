// src/layers/adaptive_activation/cpu/mod.rs

use crate::compute_manager::graph::types::DynamicContext;
use crate::compute_manager::matrix_buffer::MatrixBufferHandle;
use crate::layers::buffered_context::BufferedContext;
use crate::layers::UniversalLayerBuffered;
use crate::model_plan::param_store::ParamSlice;

use super::super::adaptive_activation::AdaptivePerFeatureActivation;

// Максимальное количество поддерживаемых базовых активаций в CPU-реализации.
const MAX_ACTIVATIONS: usize = 4;

fn activation_value(idx: usize, x: f32) -> f32 {
    match idx {
        0 => x.max(0.0), // ReLU
        1 => {
            // GELU (сигмоидальная аппроксимация)
            let sig = 1.0 / (1.0 + (-1.702 * x).exp());
            x * sig
        }
        2 => {
            // SiLU / Swish
            let sig = 1.0 / (1.0 + (-x).exp());
            x * sig
        }
        3 => x.tanh(), // Tanh
        _ => x.max(0.0),
    }
}

fn activation_derivative(idx: usize, x: f32) -> f32 {
    match idx {
        0 => {
            if x > 0.0 { 1.0 } else { 0.0 }
        }
        1 => {
            let sig = 1.0 / (1.0 + (-1.702 * x).exp());
            sig + x * sig * (1.0 - sig) * 1.702
        }
        2 => {
            let sig = 1.0 / (1.0 + (-x).exp());
            sig + x * sig * (1.0 - sig)
        }
        3 => {
            let t = x.tanh();
            1.0 - t * t
        }
        _ => {
            if x > 0.0 { 1.0 } else { 0.0 }
        }
    }
}

impl UniversalLayerBuffered for AdaptivePerFeatureActivation {
    fn forward_buffered(
        &self,
        input: &MatrixBufferHandle,
        output: &MatrixBufferHandle,
        params: &MatrixBufferHandle,
        slice: &ParamSlice,
    ) {
        assert!(
            self.num_activations <= MAX_ACTIVATIONS,
            "AdaptivePerFeatureActivation CPU: num_activations > {} not supported",
            MAX_ACTIVATIONS
        );

        let rows = input.rows();
        let cols = input.cols();
        debug_assert_eq!(cols, self.in_features);
        // Проверяем, что слайс помещается в общий буфер параметров.
        debug_assert!(
            slice.start + self.param_len() <= params.rows() * params.cols(),
            "AdaptivePerFeatureActivation: parameter slice out of bounds"
        );

        let ids = [input.id(), output.id(), params.id()];
        input.memory().write().unwrap().with_cpu_slices_mut(&ids, |slices| {
            let (first, rest) = slices.split_at_mut(1);
            let x: &[f32] = &*first[0];
            let (second, rest) = rest.split_at_mut(1);
            let y: &mut [f32] = &mut *second[0];
            let p: &[f32] = &*rest[0];

            let base = slice.start;
            let features = self.in_features;
            let num_act = self.num_activations;

            for c in 0..cols {
                // вычисляем softmax логитов для признака c
                let mut exp_sum = 0.0f32;
                let mut w = [0.0f32; MAX_ACTIVATIONS];
                for k in 0..num_act {
                    let l = p[base + k * features + c];
                    let e = l.exp();
                    w[k] = e;
                    exp_sum += e;
                }
                for k in 0..num_act {
                    w[k] /= exp_sum;
                }

                for r in 0..rows {
                    let idx = c * rows + r; // column-major
                    let x_val = x[idx];
                    let mut y_val = 0.0;
                    for k in 0..num_act {
                        y_val += w[k] * activation_value(k, x_val);
                    }
                    y[idx] = y_val;
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
        assert!(
            self.num_activations <= MAX_ACTIVATIONS,
            "AdaptivePerFeatureActivation CPU: num_activations > {} not supported",
            MAX_ACTIVATIONS
        );

        let DynamicContext::Buffered(bc) = ctx;
        let input_handle = match bc {
            BufferedContext::AdaptiveActivation { input } => input,
            _ => panic!("Expected AdaptiveActivation context"),
        };

        let rows = grad_output.rows();
        let cols = grad_output.cols();
        debug_assert_eq!(cols, self.in_features);
        debug_assert_eq!(rows, input_handle.rows());
        // Проверяем, что слайсы помещаются в буферы параметров и градиентов.
        debug_assert!(
            slice.start + self.param_len() <= params.rows() * params.cols(),
            "AdaptivePerFeatureActivation backward: parameter slice out of bounds"
        );
        debug_assert!(
            slice.start + self.param_len() <= grad_params.rows() * grad_params.cols(),
            "AdaptivePerFeatureActivation backward: grad parameter slice out of bounds"
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
                let features = self.in_features;
                let num_act = self.num_activations;
                let param_len = num_act * features;

                // локальный аккумулятор для градиентов по логитам
                let mut grad_logits = vec![0.0f32; param_len];

                for c in 0..cols {
                    // softmax логитов
                    let mut exp_sum = 0.0f32;
                    let mut w = [0.0f32; MAX_ACTIVATIONS];
                    for k in 0..num_act {
                        let l = p[base + k * features + c];
                        let e = l.exp();
                        w[k] = e;
                        exp_sum += e;
                    }
                    for k in 0..num_act {
                        w[k] /= exp_sum;
                    }

                    for r in 0..rows {
                        let idx = c * rows + r;
                        let x_val = x[idx];
                        let gout = go[idx];

                        // пересчитываем выход для градиента по логитам
                        let mut y_val = 0.0;
                        for k in 0..num_act {
                            y_val += w[k] * activation_value(k, x_val);
                        }

                        // градиент по входу
                        let mut sum_der = 0.0;
                        for k in 0..num_act {
                            sum_der += w[k] * activation_derivative(k, x_val);
                        }
                        gi[idx] = gout * sum_der;

                        // градиенты по логитам
                        for k in 0..num_act {
                            let d_l = gout * (activation_value(k, x_val) - y_val) * w[k];
                            grad_logits[k * features + c] += d_l;
                        }
                    }
                }

                // записываем накопленные градиенты по логитам в общий буфер
                for i in 0..param_len {
                    gp[base + i] = grad_logits[i];
                }
            });
    }

    fn param_len(&self) -> usize {
        self.in_features * self.num_activations
    }

    fn input_features(&self) -> usize {
        self.in_features
    }

    fn output_features(&self) -> usize {
        self.in_features
    }
}