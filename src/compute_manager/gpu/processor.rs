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
    params: &[f32],
    input: MatrixBufferHandle,
) -> (MatrixBufferHandle, Vec<DynamicContext>) {
    assert!(input.is_gpu(), "Input must be GPU handle");
    let mut current = input;
    let mut ctxs = Vec::with_capacity(layers.len());

    for (layer, slice) in layers.iter().zip(slices.iter()) {
        if let Some(linear) = layer.as_linear() {
            let in_feat = linear.input_features();
            let out_feat = linear.output_features();
            let w_start = slice.start;
            let b_start = w_start + in_feat * out_feat;

            // Загружаем веса на GPU как дескриптор
            let weight_vec = &params[w_start..w_start + in_feat * out_feat];
            let weight_gpu = gpu_compute.upload_vec_to_gpu_handle(weight_vec, out_feat, in_feat);
            let bias = &params[b_start..b_start + out_feat];

            let out_handle = gpu_compute.allocate_gpu_matrix_handle(current.rows(), out_feat);
            gpu_compute.run_linear_forward_buffered_handle(
                &current,
                &weight_gpu,
                bias,
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
            let thresholds = &params[slice.start..slice.start + soft_sparse.in_features];
            let out_handle = gpu_compute.allocate_gpu_matrix_handle(current.rows(), current.cols());
            gpu_compute.run_softsparse_forward_buffered_handle(
                &current,
                thresholds,
                soft_sparse.temperature,
                &out_handle,
            );
            ctxs.push(DynamicContext::Buffered(BufferedContext::SoftSparseGate {
                input: current.clone(),
            }));
            current = out_handle;
        } else if let Some(soft_keep) = layer.as_soft_keep_gate() {
            let thresholds = &params[slice.start..slice.start + soft_keep.in_features];
            let out_handle = gpu_compute.allocate_gpu_matrix_handle(current.rows(), current.cols());
            gpu_compute.run_softkeep_forward_buffered_handle(
                &current,
                thresholds,
                soft_keep.temperature,
                &out_handle,
            );
            ctxs.push(DynamicContext::Buffered(BufferedContext::SoftKeepGate {
                input: current.clone(),
            }));
            current = out_handle;
        } else if let Some(dual) = layer.as_dual_anchor() {
            let features = dual.features;
            let min_vals = &params[slice.start..slice.start + features];
            let max_vals = &params[slice.start + features..slice.start + 2 * features];
            let alpha = params[slice.start + 2 * features];
            let out_handle = gpu_compute.allocate_gpu_matrix_handle(current.rows(), current.cols());
            gpu_compute.run_dualanchor_forward_buffered_handle(
                &current,
                min_vals,
                max_vals,
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
pub fn process_backward_gpu_buffered(
    gpu_compute: &GpuCompute,
    layers: &[Box<dyn UniversalLayer>],
    slices: &[ParamSlice],
    contexts: &[DynamicContext],
    params: &[f32],
    grad_output: MatrixBufferHandle,
    total_grad: &mut [f32],
) -> MatrixBufferHandle {
    assert!(grad_output.is_gpu(), "Grad output must be GPU handle");
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

            let weight_vec = &params[w_start..w_start + in_feat * out_feat];
            let weight_gpu = gpu_compute.upload_vec_to_gpu_handle(weight_vec, out_feat, in_feat);

            let grad_input_handle = gpu_compute.allocate_gpu_matrix_handle(current_grad.rows(), in_feat);
            let grad_weight_handle = gpu_compute.allocate_gpu_matrix_handle(out_feat, in_feat);
            let grad_bias_handle = gpu_compute.allocate_gpu_matrix_handle(1, out_feat);

            let grad_bias = gpu_compute.run_linear_backward_buffered_handle(
                &input_handle,
                &weight_gpu,
                &current_grad,
                &grad_input_handle,
                &grad_weight_handle,
                &grad_bias_handle,
            );

            let grad_weight_vec = gpu_compute.download_gpu_handle_to_vec(&grad_weight_handle);
            for (i, &g) in grad_weight_vec.iter().enumerate() {
                total_grad[w_start + i] += g;
            }
            for (i, &g) in grad_bias.iter().enumerate() {
                total_grad[b_start + i] += g;
            }

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
            let thresholds = &params[slice.start..slice.start + soft_sparse.in_features];
            let grad_input_handle = gpu_compute.allocate_gpu_matrix_handle(current_grad.rows(), current_grad.cols());
            let grad_thresh_handle = gpu_compute.allocate_gpu_matrix_handle(1, soft_sparse.in_features);
            let grad_thresh_vec = gpu_compute.run_softsparse_backward_buffered_handle(
                &input_handle,
                &current_grad,
                thresholds,
                soft_sparse.temperature,
                &grad_input_handle,
                &grad_thresh_handle,
            );
            for (i, &g) in grad_thresh_vec.iter().enumerate() {
                total_grad[slice.start + i] += g;
            }
            current_grad = grad_input_handle;
        } else if let Some(soft_keep) = layer.as_soft_keep_gate() {
            let DynamicContext::Buffered(bc) = ctx;
            let input_handle = match bc {
                BufferedContext::SoftKeepGate { input } => input.clone(),
                _ => panic!("Expected SoftKeepGate Buffered context"),
            };
            let thresholds = &params[slice.start..slice.start + soft_keep.in_features];
            let grad_input_handle = gpu_compute.allocate_gpu_matrix_handle(current_grad.rows(), current_grad.cols());
            let grad_thresh_handle = gpu_compute.allocate_gpu_matrix_handle(1, soft_keep.in_features);
            let grad_thresh_vec = gpu_compute.run_softkeep_backward_buffered_handle(
                &input_handle,
                &current_grad,
                thresholds,
                soft_keep.temperature,
                &grad_input_handle,
                &grad_thresh_handle,
            );
            for (i, &g) in grad_thresh_vec.iter().enumerate() {
                total_grad[slice.start + i] += g;
            }
            current_grad = grad_input_handle;
        } else if let Some(dual) = layer.as_dual_anchor() {
            let DynamicContext::Buffered(bc) = ctx;
            let input_handle = match bc {
                BufferedContext::DualAnchor1D { input } => input.clone(),
                _ => panic!("Expected DualAnchor1D Buffered context"),
            };
            let features = dual.features;
            let min_vals = &params[slice.start..slice.start + features];
            let max_vals = &params[slice.start + features..slice.start + 2 * features];
            let alpha = params[slice.start + 2 * features];
            let grad_input_handle = gpu_compute.allocate_gpu_matrix_handle(current_grad.rows(), current_grad.cols());
            let grad_min_handle = gpu_compute.allocate_gpu_matrix_handle(1, features);
            let grad_max_handle = gpu_compute.allocate_gpu_matrix_handle(1, features);
            let grad_alpha_handle = gpu_compute.allocate_gpu_matrix_handle(1, 1);
            let grad_vec = gpu_compute.run_dualanchor_backward_buffered_handle(
                &input_handle,
                &current_grad,
                min_vals,
                max_vals,
                alpha,
                &grad_input_handle,
                &grad_min_handle,
                &grad_max_handle,
                &grad_alpha_handle,
            );
            for (i, &g) in grad_vec.iter().enumerate() {
                total_grad[slice.start + i] += g;
            }
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