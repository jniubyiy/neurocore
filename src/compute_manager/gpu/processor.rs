// src/compute_manager/gpu/processor.rs

use crate::compute_manager::graph::types::DynamicContext;
use crate::compute_manager::matrix_buffer::MatrixBufferHandle;
use crate::layers::buffered_context::BufferedContext;
use crate::layers::UniversalLayer;
use crate::model_plan::param_store::ParamSlice;

use super::compute::GpuCompute;

/// Прямой проход на GPU с использованием MatrixBufferHandle.
/// Вход и выход — GPU-дескрипторы. Контексты создаются как Buffered.
pub fn process_forward_gpu_buffered(
    gpu_compute: &GpuCompute,
    layers: &[Box<dyn UniversalLayer>],
    slices: &[ParamSlice],
    params: &MatrixBufferHandle,
    input: MatrixBufferHandle,
) -> (MatrixBufferHandle, Vec<DynamicContext>) {
    assert!(input.is_gpu(), "Input must be GPU handle");
    assert!(!params.is_gpu(), "process_forward_gpu_buffered: params must be CPU handle");

    let mut current = input;
    let mut ctxs = Vec::with_capacity(layers.len());

    for (layer, slice) in layers.iter().zip(slices.iter()) {
        if let Some(linear) = layer.as_linear() {
            let in_feat = linear.input_features();
            let out_feat = linear.output_features();
            let w_start = slice.start;
            let b_start = w_start + in_feat * out_feat;

            // Читаем только веса и bias текущего слоя
            let weight_vec = params.read_range(w_start, in_feat * out_feat);
            let bias = params.read_range(b_start, out_feat);
            let weight_gpu = gpu_compute.upload_vec_to_gpu_handle(&weight_vec, out_feat, in_feat);

            let out_handle = gpu_compute.allocate_gpu_matrix_handle(current.rows(), out_feat);
            gpu_compute.run_linear_forward_buffered_handle(
                &current,
                &weight_gpu,
                &bias,
                &out_handle,
            );

            let input_for_ctx = current.clone();
            ctxs.push(DynamicContext::Buffered(BufferedContext::Linear {
                input: input_for_ctx,
            }));
            current = out_handle;
        } else if let Some(_) = layer.as_relu() {
            let out_handle = gpu_compute.allocate_gpu_matrix_handle(current.rows(), current.cols());
            gpu_compute.run_relu_forward_buffered_handle(&current, &out_handle);
            ctxs.push(DynamicContext::Buffered(BufferedContext::ReLU {
                input: current.clone(),
            }));
            current = out_handle;
        } else if let Some(_) = layer.as_sigmoid() {
            let out_handle = gpu_compute.allocate_gpu_matrix_handle(current.rows(), current.cols());
            gpu_compute.run_sigmoid_forward_buffered_handle(&current, &out_handle);
            ctxs.push(DynamicContext::Buffered(BufferedContext::Sigmoid {
                output: out_handle.clone(),
            }));
            current = out_handle;
        } else if let Some(_) = layer.as_tanh() {
            let out_handle = gpu_compute.allocate_gpu_matrix_handle(current.rows(), current.cols());
            gpu_compute.run_tanh_forward_buffered_handle(&current, &out_handle);
            ctxs.push(DynamicContext::Buffered(BufferedContext::Tanh {
                output: out_handle.clone(),
            }));
            current = out_handle;
        } else if let Some(leaky) = layer.as_leaky_relu() {
            let out_handle = gpu_compute.allocate_gpu_matrix_handle(current.rows(), current.cols());
            gpu_compute.run_leaky_relu_forward_buffered_handle(&current, &out_handle, leaky.alpha);
            ctxs.push(DynamicContext::Buffered(BufferedContext::LeakyReLU {
                input: current.clone(),
            }));
            current = out_handle;
        } else if let Some(_) = layer.as_softmax() {
            let out_handle = gpu_compute.allocate_gpu_matrix_handle(current.rows(), current.cols());
            gpu_compute.run_softmax_forward_buffered_handle(&current, &out_handle);
            ctxs.push(DynamicContext::Buffered(BufferedContext::Softmax {
                output: out_handle.clone(),
            }));
            current = out_handle;
        } else if let Some(_) = layer.as_identity() {
            let out_handle = gpu_compute.allocate_gpu_matrix_handle(current.rows(), current.cols());
            gpu_compute.copy_gpu_handle_to_gpu_handle(&current, &out_handle);
            ctxs.push(DynamicContext::Buffered(BufferedContext::Identity {
                input: current.clone(),
            }));
            current = out_handle;
        } else if let Some(memory) = layer.as_memory() {
            let out_handle = gpu_compute.allocate_gpu_matrix_handle(current.rows(), current.cols());
            gpu_compute.run_memory_forward_buffered_handle(&current, &out_handle, memory.alpha);
            ctxs.push(DynamicContext::Buffered(BufferedContext::Memory {
                input: current.clone(),
            }));
            current = out_handle;
        } else if let Some(soft_sparse) = layer.as_soft_sparse_gate() {
            let thresholds = params.read_range(slice.start, soft_sparse.in_features);
            let out_handle = gpu_compute.allocate_gpu_matrix_handle(current.rows(), current.cols());
            gpu_compute.run_softsparse_forward_buffered_handle(
                &current,
                &thresholds,
                soft_sparse.temperature,
                &out_handle,
            );
            ctxs.push(DynamicContext::Buffered(BufferedContext::SoftSparseGate {
                input: current.clone(),
            }));
            current = out_handle;
        } else if let Some(soft_keep) = layer.as_soft_keep_gate() {
            let thresholds = params.read_range(slice.start, soft_keep.in_features);
            let out_handle = gpu_compute.allocate_gpu_matrix_handle(current.rows(), current.cols());
            gpu_compute.run_softkeep_forward_buffered_handle(
                &current,
                &thresholds,
                soft_keep.temperature,
                &out_handle,
            );
            ctxs.push(DynamicContext::Buffered(BufferedContext::SoftKeepGate {
                input: current.clone(),
            }));
            current = out_handle;
        } else if let Some(dual) = layer.as_dual_anchor() {
            let features = dual.features;
            let min_vals = params.read_range(slice.start, features);
            let max_vals = params.read_range(slice.start + features, features);
            let alpha_vec = params.read_range(slice.start + 2 * features, 1);
            let alpha = alpha_vec[0];

            let out_handle = gpu_compute.allocate_gpu_matrix_handle(current.rows(), current.cols());
            gpu_compute.run_dualanchor_forward_buffered_handle(
                &current,
                &min_vals,
                &max_vals,
                alpha,
                &out_handle,
            );
            ctxs.push(DynamicContext::Buffered(BufferedContext::DualAnchor1D {
                input: current.clone(),
            }));
            current = out_handle;
        } else {
            panic!(
                "Unsupported layer in GPU buffered forward: {:?}",
                std::any::type_name_of_val(layer.as_ref())
            );
        }
    }

    (current, ctxs)
}

/// Обратный проход на GPU с использованием MatrixBufferHandle.
/// Входной градиент — GPU-дескриптор, выходной градиент — GPU-дескриптор.
/// Градиенты параметров записываются напрямую в глобальный буфер `grad_params_handle`.
pub fn process_backward_gpu_buffered(
    gpu_compute: &GpuCompute,
    layers: &[Box<dyn UniversalLayer>],
    slices: &[ParamSlice],
    contexts: &[DynamicContext],
    params: &MatrixBufferHandle,
    grad_output: MatrixBufferHandle,
    grad_params_handle: &MatrixBufferHandle,
) -> MatrixBufferHandle {
    assert!(grad_output.is_gpu(), "Grad output must be GPU handle");
    assert!(!params.is_gpu(), "process_backward_gpu_buffered: params must be CPU handle");

    let num_layers = layers.len();
    assert_eq!(contexts.len(), num_layers);
    assert_eq!(slices.len(), num_layers);

    let mut current_grad = grad_output;

    for idx in (0..num_layers).rev() {
        let layer = &layers[idx];
        let slice = &slices[idx];
        let ctx = &contexts[idx];

        if let Some(linear) = layer.as_linear() {
            let in_feat = linear.input_features();
            let out_feat = linear.output_features();
            let w_start = slice.start;
            let b_start = w_start + in_feat * out_feat;

            let DynamicContext::Buffered(bc) = ctx;
            let input_handle = match bc {
                BufferedContext::Linear { input } => input.clone(),
                _ => panic!("Expected Linear Buffered context"),
            };

            // Читаем только веса текущего слоя
            let weight_vec = params.read_range(w_start, in_feat * out_feat);
            let weight_gpu = gpu_compute.upload_vec_to_gpu_handle(&weight_vec, out_feat, in_feat);

            let grad_input_handle = gpu_compute.allocate_gpu_matrix_handle(current_grad.rows(), in_feat);
            let grad_weight_handle = gpu_compute.allocate_gpu_matrix_handle(out_feat, in_feat);
            let grad_bias_handle = gpu_compute.allocate_gpu_matrix_handle(1, out_feat);

            // Выделяем CPU-буферы для градиентов весов и смещений
            let grad_weight_cpu = gpu_compute.allocate_cpu_matrix_handle(out_feat, in_feat);
            let grad_bias_cpu = gpu_compute.allocate_cpu_matrix_handle(1, out_feat);

            gpu_compute.run_linear_backward_buffered_handle(
                &input_handle,
                &weight_gpu,
                &current_grad,
                &grad_input_handle,
                &grad_weight_handle,
                &grad_bias_handle,
                &grad_bias_cpu,
            );

            // Копируем градиенты весов из GPU в CPU
            gpu_compute.copy_gpu_to_cpu_handle(&grad_weight_handle, &grad_weight_cpu);

            // Переносим данные в общий буфер градиентов
            let weight_data = grad_weight_cpu.read().as_slice().unwrap().to_vec();
            let bias_data = grad_bias_cpu.read().as_slice().unwrap().to_vec();
            grad_params_handle.with_cpu_data_mut(|grad_data| {
                grad_data[w_start..w_start + weight_data.len()].copy_from_slice(&weight_data);
                grad_data[b_start..b_start + bias_data.len()].copy_from_slice(&bias_data);
            });

            current_grad = grad_input_handle;
        } else if let Some(_) = layer.as_relu() {
            let DynamicContext::Buffered(bc) = ctx;
            let input_handle = match bc {
                BufferedContext::ReLU { input } => input.clone(),
                _ => panic!("Expected ReLU Buffered context"),
            };
            let grad_input_handle = gpu_compute.allocate_gpu_matrix_handle(current_grad.rows(), current_grad.cols());
            gpu_compute.run_relu_backward_buffered_handle(
                &input_handle,
                &current_grad,
                &grad_input_handle,
            );
            current_grad = grad_input_handle;
        } else if let Some(_) = layer.as_sigmoid() {
            let DynamicContext::Buffered(bc) = ctx;
            let output_handle = match bc {
                BufferedContext::Sigmoid { output } => output.clone(),
                _ => panic!("Expected Sigmoid Buffered context"),
            };
            let grad_input_handle = gpu_compute.allocate_gpu_matrix_handle(current_grad.rows(), current_grad.cols());
            gpu_compute.run_sigmoid_backward_buffered_handle(
                &output_handle,
                &current_grad,
                &grad_input_handle,
            );
            current_grad = grad_input_handle;
        } else if let Some(_) = layer.as_tanh() {
            let DynamicContext::Buffered(bc) = ctx;
            let output_handle = match bc {
                BufferedContext::Tanh { output } => output.clone(),
                _ => panic!("Expected Tanh Buffered context"),
            };
            let grad_input_handle = gpu_compute.allocate_gpu_matrix_handle(current_grad.rows(), current_grad.cols());
            gpu_compute.run_tanh_backward_buffered_handle(
                &output_handle,
                &current_grad,
                &grad_input_handle,
            );
            current_grad = grad_input_handle;
        } else if let Some(leaky) = layer.as_leaky_relu() {
            let DynamicContext::Buffered(bc) = ctx;
            let input_handle = match bc {
                BufferedContext::LeakyReLU { input } => input.clone(),
                _ => panic!("Expected LeakyReLU Buffered context"),
            };
            let grad_input_handle = gpu_compute.allocate_gpu_matrix_handle(current_grad.rows(), current_grad.cols());
            gpu_compute.run_leaky_relu_backward_buffered_handle(
                &input_handle,
                &current_grad,
                &grad_input_handle,
                leaky.alpha,
            );
            current_grad = grad_input_handle;
        } else if let Some(_) = layer.as_softmax() {
            let DynamicContext::Buffered(bc) = ctx;
            let output_handle = match bc {
                BufferedContext::Softmax { output } => output.clone(),
                _ => panic!("Expected Softmax Buffered context"),
            };
            let grad_input_handle = gpu_compute.allocate_gpu_matrix_handle(current_grad.rows(), current_grad.cols());
            gpu_compute.run_softmax_backward_buffered_handle(
                &output_handle,
                &current_grad,
                &grad_input_handle,
            );
            current_grad = grad_input_handle;
        } else if let Some(_) = layer.as_identity() {
            let grad_input_handle = gpu_compute.allocate_gpu_matrix_handle(current_grad.rows(), current_grad.cols());
            gpu_compute.copy_gpu_handle_to_gpu_handle(&current_grad, &grad_input_handle);
            current_grad = grad_input_handle;
        } else if let Some(memory) = layer.as_memory() {
            let grad_input_handle = gpu_compute.allocate_gpu_matrix_handle(current_grad.rows(), current_grad.cols());
            gpu_compute.run_memory_backward_buffered_handle(
                &current_grad,
                &grad_input_handle,
                memory.alpha,
            );
            current_grad = grad_input_handle;
        } else if let Some(soft_sparse) = layer.as_soft_sparse_gate() {
            let DynamicContext::Buffered(bc) = ctx;
            let input_handle = match bc {
                BufferedContext::SoftSparseGate { input } => input.clone(),
                _ => panic!("Expected SoftSparseGate Buffered context"),
            };
            let thresholds = params.read_range(slice.start, soft_sparse.in_features);
            let grad_input_handle = gpu_compute.allocate_gpu_matrix_handle(current_grad.rows(), current_grad.cols());
            let grad_thresh_handle = gpu_compute.allocate_gpu_matrix_handle(1, soft_sparse.in_features);
            let grad_thresh_cpu = gpu_compute.allocate_cpu_matrix_handle(1, soft_sparse.in_features);

            gpu_compute.run_softsparse_backward_buffered_handle(
                &input_handle,
                &current_grad,
                &thresholds,
                soft_sparse.temperature,
                &grad_input_handle,
                &grad_thresh_handle,
                &grad_thresh_cpu,
            );

            // Переносим данные в общий буфер
            let thresh_data = grad_thresh_cpu.read().as_slice().unwrap().to_vec();
            grad_params_handle.with_cpu_data_mut(|grad_data| {
                grad_data[slice.start..slice.start + thresh_data.len()].copy_from_slice(&thresh_data);
            });

            current_grad = grad_input_handle;
        } else if let Some(soft_keep) = layer.as_soft_keep_gate() {
            let DynamicContext::Buffered(bc) = ctx;
            let input_handle = match bc {
                BufferedContext::SoftKeepGate { input } => input.clone(),
                _ => panic!("Expected SoftKeepGate Buffered context"),
            };
            let thresholds = params.read_range(slice.start, soft_keep.in_features);
            let grad_input_handle = gpu_compute.allocate_gpu_matrix_handle(current_grad.rows(), current_grad.cols());
            let grad_thresh_handle = gpu_compute.allocate_gpu_matrix_handle(1, soft_keep.in_features);
            let grad_thresh_cpu = gpu_compute.allocate_cpu_matrix_handle(1, soft_keep.in_features);

            gpu_compute.run_softkeep_backward_buffered_handle(
                &input_handle,
                &current_grad,
                &thresholds,
                soft_keep.temperature,
                &grad_input_handle,
                &grad_thresh_handle,
                &grad_thresh_cpu,
            );

            // Переносим данные в общий буфер
            let thresh_data = grad_thresh_cpu.read().as_slice().unwrap().to_vec();
            grad_params_handle.with_cpu_data_mut(|grad_data| {
                grad_data[slice.start..slice.start + thresh_data.len()].copy_from_slice(&thresh_data);
            });

            current_grad = grad_input_handle;
        } else if let Some(dual) = layer.as_dual_anchor() {
            let DynamicContext::Buffered(bc) = ctx;
            let input_handle = match bc {
                BufferedContext::DualAnchor1D { input } => input.clone(),
                _ => panic!("Expected DualAnchor1D Buffered context"),
            };
            let features = dual.features;
            let min_vals = params.read_range(slice.start, features);
            let max_vals = params.read_range(slice.start + features, features);
            let alpha_vec = params.read_range(slice.start + 2 * features, 1);
            let alpha = alpha_vec[0];

            let grad_input_handle = gpu_compute.allocate_gpu_matrix_handle(current_grad.rows(), current_grad.cols());
            let grad_min_handle = gpu_compute.allocate_gpu_matrix_handle(1, features);
            let grad_max_handle = gpu_compute.allocate_gpu_matrix_handle(1, features);
            let grad_alpha_handle = gpu_compute.allocate_gpu_matrix_handle(1, 1);

            let grad_min_cpu = gpu_compute.allocate_cpu_matrix_handle(1, features);
            let grad_max_cpu = gpu_compute.allocate_cpu_matrix_handle(1, features);
            let grad_alpha_cpu = gpu_compute.allocate_cpu_matrix_handle(1, 1);

            gpu_compute.run_dualanchor_backward_buffered_handle(
                &input_handle,
                &current_grad,
                &min_vals,
                &max_vals,
                alpha,
                &grad_input_handle,
                &grad_min_handle,
                &grad_max_handle,
                &grad_alpha_handle,
                &grad_min_cpu,
                &grad_max_cpu,
                &grad_alpha_cpu,
            );

            // Переносим данные в общий буфер
            let min_data = grad_min_cpu.read().as_slice().unwrap().to_vec();
            let max_data = grad_max_cpu.read().as_slice().unwrap().to_vec();
            let alpha_data = grad_alpha_cpu.read().as_slice().unwrap().to_vec();
            grad_params_handle.with_cpu_data_mut(|grad_data| {
                grad_data[slice.start..slice.start + features].copy_from_slice(&min_data);
                grad_data[slice.start + features..slice.start + 2 * features].copy_from_slice(&max_data);
                grad_data[slice.start + 2 * features] = alpha_data[0];
            });

            current_grad = grad_input_handle;
        } else {
            panic!(
                "Unsupported layer in GPU buffered backward: {:?}",
                std::any::type_name_of_val(layer.as_ref())
            );
        }
    }

    current_grad
}