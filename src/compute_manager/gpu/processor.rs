// src/compute_manager/gpu/processor.rs

use crate::compute_manager::graph::types::DynamicContext;
use crate::compute_manager::matrix_buffer::view::MatrixBufferView;
use crate::compute_manager::matrix_buffer::MatrixBufferHandle;
use crate::layers::buffered_context::BufferedContext;
use crate::layers::UniversalLayer;
use crate::model_plan::param_store::ParamSlice;

use super::compute::GpuCompute;

/// Прямой проход на GPU с использованием MatrixBufferHandle.
/// Вход и выход — GPU-дескрипторы. Контексты создаются как Buffered.
///
/// Параметры сегмента уже должны находиться на GPU (в `params_handle`).
/// Доступ к отдельным слоям осуществляется через `MatrixBufferView`,
/// который представляет собой непрерывный диапазон внутри буфера.
pub fn process_forward_gpu_buffered(
    gpu_compute: &GpuCompute,
    layers: &[Box<dyn UniversalLayer>],
    slices: &[ParamSlice],
    params_handle: &MatrixBufferHandle,
    input: MatrixBufferHandle,
) -> (MatrixBufferHandle, Vec<DynamicContext>) {
    assert!(input.is_gpu(), "Input must be GPU handle");
    assert!(
        params_handle.is_gpu() || params_handle.rows() == 0,
        "process_forward_gpu_buffered: params must be GPU or empty for parameterless segment"
    );

    let mut current = input;
    let mut ctxs = Vec::with_capacity(layers.len());
    let mut memory_idx = 0usize; // счётчик слоёв Memory

    for (layer, slice) in layers.iter().zip(slices.iter()) {
        if let Some(linear) = layer.as_linear() {
            let in_feat = linear.input_features();
            let out_feat = linear.output_features();
            let w_start = slice.start;
            let b_start = w_start + in_feat * out_feat;

            // Создаём view для весов и смещений из общего GPU-буфера параметров.
            let weight_view = MatrixBufferView::with_shape(
                params_handle.clone(),
                w_start,
                in_feat * out_feat,
                out_feat,
                in_feat,
            );
            let bias_view = MatrixBufferView::with_shape(
                params_handle.clone(),
                b_start,
                out_feat,
                1,
                out_feat,
            );

            let out_handle = gpu_compute.allocate_gpu_matrix_handle(current.rows(), out_feat);
            gpu_compute.run_linear_forward_buffered_handle(
                &current,
                &weight_view,
                &bias_view,
                &out_handle,
            );

            ctxs.push(DynamicContext::Buffered(BufferedContext::Linear {
                input: current.clone(),
            }));
            current = out_handle;
        } else if layer.as_relu().is_some() {
            let out_handle = gpu_compute.allocate_gpu_matrix_handle(current.rows(), current.cols());
            gpu_compute.run_relu_forward_buffered_handle(&current, &out_handle);
            ctxs.push(DynamicContext::Buffered(BufferedContext::ReLU {
                input: current.clone(),
            }));
            current = out_handle;
        } else if layer.as_sigmoid().is_some() {
            let out_handle = gpu_compute.allocate_gpu_matrix_handle(current.rows(), current.cols());
            gpu_compute.run_sigmoid_forward_buffered_handle(&current, &out_handle);
            ctxs.push(DynamicContext::Buffered(BufferedContext::Sigmoid {
                output: out_handle.clone(),
            }));
            current = out_handle;
        } else if layer.as_tanh().is_some() {
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
        } else if layer.as_softmax().is_some() {
            let out_handle = gpu_compute.allocate_gpu_matrix_handle(current.rows(), current.cols());
            gpu_compute.run_softmax_forward_buffered_handle(&current, &out_handle);
            ctxs.push(DynamicContext::Buffered(BufferedContext::Softmax {
                output: out_handle.clone(),
            }));
            current = out_handle;
        } else if layer.as_identity().is_some() {
            let out_handle = gpu_compute.allocate_gpu_matrix_handle(current.rows(), current.cols());
            gpu_compute.run_identity_forward_buffered_handle(&current, &out_handle);
            ctxs.push(DynamicContext::Buffered(BufferedContext::Identity {
                input: current.clone(),
            }));
            current = out_handle;
        } else if let Some(memory) = layer.as_memory() {
            let out_handle = gpu_compute.allocate_gpu_matrix_handle(current.rows(), current.cols());
            gpu_compute.run_memory_forward_buffered_handle(
                &current,
                &out_handle,
                memory.alpha,
                memory_idx,
            );
            memory_idx += 1;
            ctxs.push(DynamicContext::Buffered(BufferedContext::Memory {
                input: current.clone(),
            }));
            current = out_handle;
        } else if let Some(soft_sparse) = layer.as_soft_sparse_gate() {
            let features = soft_sparse.in_features;
            let thresholds_view = MatrixBufferView::new(
                params_handle.clone(),
                slice.start,
                features,
            );

            let out_handle = gpu_compute.allocate_gpu_matrix_handle(current.rows(), current.cols());
            gpu_compute.run_softsparse_forward_buffered_handle(
                &current,
                &thresholds_view,
                soft_sparse.temperature,
                &out_handle,
            );
            ctxs.push(DynamicContext::Buffered(BufferedContext::SoftSparseGate {
                input: current.clone(),
            }));
            current = out_handle;
        } else if let Some(soft_keep) = layer.as_soft_keep_gate() {
            let features = soft_keep.in_features;
            let thresholds_view = MatrixBufferView::new(
                params_handle.clone(),
                slice.start,
                features,
            );

            let out_handle = gpu_compute.allocate_gpu_matrix_handle(current.rows(), current.cols());
            gpu_compute.run_softkeep_forward_buffered_handle(
                &current,
                &thresholds_view,
                soft_keep.temperature,
                &out_handle,
            );
            ctxs.push(DynamicContext::Buffered(BufferedContext::SoftKeepGate {
                input: current.clone(),
            }));
            current = out_handle;
        } else if let Some(dual) = layer.as_dual_anchor() {
            let features = dual.features;
            let min_view = MatrixBufferView::new(params_handle.clone(), slice.start, features);
            let max_view = MatrixBufferView::new(
                params_handle.clone(),
                slice.start + features,
                features,
            );
            let alpha_view = MatrixBufferView::new(
                params_handle.clone(),
                slice.start + 2 * features,
                1,
            );

            let out_handle = gpu_compute.allocate_gpu_matrix_handle(current.rows(), current.cols());
            gpu_compute.run_dualanchor_forward_buffered_handle(
                &current,
                &min_view,
                &max_view,
                &alpha_view,
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
/// Градиенты параметров записываются напрямую в `grad_params_handle` (GPU).
pub fn process_backward_gpu_buffered(
    gpu_compute: &GpuCompute,
    layers: &[Box<dyn UniversalLayer>],
    slices: &[ParamSlice],
    contexts: &[DynamicContext],
    params_handle: &MatrixBufferHandle,
    grad_output: MatrixBufferHandle,
    grad_params_handle: &MatrixBufferHandle,
) -> MatrixBufferHandle {
    assert!(grad_output.is_gpu(), "Grad output must be GPU handle");
    assert!(
        grad_params_handle.is_gpu() || grad_params_handle.rows() == 0,
        "grad_params_handle must be GPU or empty"
    );

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

            let weight_view = MatrixBufferView::with_shape(
                params_handle.clone(),
                w_start,
                in_feat * out_feat,
                out_feat,
                in_feat,
            );

            let grad_weight_view = MatrixBufferView::with_shape(
                grad_params_handle.clone(),
                w_start,
                in_feat * out_feat,
                out_feat,
                in_feat,
            );
            let grad_bias_view = MatrixBufferView::with_shape(
                grad_params_handle.clone(),
                b_start,
                out_feat,
                1,
                out_feat,
            );

            let grad_input_handle = gpu_compute.allocate_gpu_matrix_handle(current_grad.rows(), in_feat);

            gpu_compute.run_linear_backward_buffered_handle(
                &input_handle,
                &weight_view,
                &current_grad,
                &grad_input_handle,
                &grad_weight_view,
                &grad_bias_view,
            );

            current_grad = grad_input_handle;
        } else if layer.as_relu().is_some() {
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
        } else if layer.as_sigmoid().is_some() {
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
        } else if layer.as_tanh().is_some() {
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
        } else if layer.as_softmax().is_some() {
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
        } else if layer.as_identity().is_some() {
            let grad_input_handle = gpu_compute.allocate_gpu_matrix_handle(current_grad.rows(), current_grad.cols());
            gpu_compute.run_identity_backward_buffered_handle(
                &current_grad,
                &grad_input_handle,
            );
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
            let features = soft_sparse.in_features;
            let DynamicContext::Buffered(bc) = ctx;
            let input_handle = match bc {
                BufferedContext::SoftSparseGate { input } => input.clone(),
                _ => panic!("Expected SoftSparseGate Buffered context"),
            };
            let thresholds_view = MatrixBufferView::new(params_handle.clone(), slice.start, features);
            let grad_thresh_view = MatrixBufferView::new(grad_params_handle.clone(), slice.start, features);

            let grad_input_handle = gpu_compute.allocate_gpu_matrix_handle(current_grad.rows(), current_grad.cols());

            gpu_compute.run_softsparse_backward_buffered_handle(
                &input_handle,
                &current_grad,
                &thresholds_view,
                soft_sparse.temperature,
                &grad_input_handle,
                &grad_thresh_view,
            );
            current_grad = grad_input_handle;
        } else if let Some(soft_keep) = layer.as_soft_keep_gate() {
            let features = soft_keep.in_features;
            let DynamicContext::Buffered(bc) = ctx;
            let input_handle = match bc {
                BufferedContext::SoftKeepGate { input } => input.clone(),
                _ => panic!("Expected SoftKeepGate Buffered context"),
            };
            let thresholds_view = MatrixBufferView::new(params_handle.clone(), slice.start, features);
            let grad_thresh_view = MatrixBufferView::new(grad_params_handle.clone(), slice.start, features);

            let grad_input_handle = gpu_compute.allocate_gpu_matrix_handle(current_grad.rows(), current_grad.cols());

            gpu_compute.run_softkeep_backward_buffered_handle(
                &input_handle,
                &current_grad,
                &thresholds_view,
                soft_keep.temperature,
                &grad_input_handle,
                &grad_thresh_view,
            );
            current_grad = grad_input_handle;
        } else if let Some(dual) = layer.as_dual_anchor() {
            let features = dual.features;
            let DynamicContext::Buffered(bc) = ctx;
            let input_handle = match bc {
                BufferedContext::DualAnchor1D { input } => input.clone(),
                _ => panic!("Expected DualAnchor1D Buffered context"),
            };
            let min_view = MatrixBufferView::new(params_handle.clone(), slice.start, features);
            let max_view = MatrixBufferView::new(
                params_handle.clone(),
                slice.start + features,
                features,
            );
            let alpha_view = MatrixBufferView::new(
                params_handle.clone(),
                slice.start + 2 * features,
                1,
            );

            let grad_min_view = MatrixBufferView::new(grad_params_handle.clone(), slice.start, features);
            let grad_max_view = MatrixBufferView::new(
                grad_params_handle.clone(),
                slice.start + features,
                features,
            );
            let grad_alpha_view = MatrixBufferView::new(
                grad_params_handle.clone(),
                slice.start + 2 * features,
                1,
            );

            let grad_input_handle = gpu_compute.allocate_gpu_matrix_handle(current_grad.rows(), current_grad.cols());

            gpu_compute.run_dualanchor_backward_buffered_handle(
                &input_handle,
                &current_grad,
                &min_view,
                &max_view,
                &alpha_view,
                &grad_input_handle,
                &grad_min_view,
                &grad_max_view,
                &grad_alpha_view,
            );
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