// src/compute_manager/gpu/processor.rs

use faer::Mat;
use crate::compute_manager::graph::types::DynamicContext;
use crate::compute_manager::persistent_buffer::SegmentPersistentBuffers;
use crate::compute_manager::matrix_buffer::MatrixBuffer;
use crate::layers::UniversalLayer;
use crate::layers::mat_context::MatContext;
use crate::model_plan::param_store::ParamSlice;
use super::compute::GpuCompute;

/// Прямой проход на GPU с persistent-буферами (старая версия, для обратной совместимости).
/// Принимает `Mat`, возвращает `Mat` и контексты.
pub fn process_forward_gpu(
    gpu_compute: &GpuCompute,
    _segment_buffers: &SegmentPersistentBuffers,
    layers: &[Box<dyn UniversalLayer>],
    slices: &[ParamSlice],
    params: &[f32],
    input: &Mat<f32>,
) -> (Mat<f32>, Vec<DynamicContext>) {
    let mut current = input.clone();
    let mut ctxs = Vec::with_capacity(layers.len());

    for (layer, slice) in layers.iter().zip(slices.iter()) {
        if let Some(linear) = layer.as_linear() {
            let (weight, bias) = linear.get_weight_matrix_and_bias(params, slice);
            let input_for_ctx = current.clone();
            current = gpu_compute.run_linear_forward(&current, &weight, &bias);
            ctxs.push(DynamicContext::Mat(MatContext::Linear {
                input: input_for_ctx,
            }));
        } else if let Some(_) = layer.as_relu() {
            let input_for_ctx = current.clone();
            current = gpu_compute.run_relu_forward(&current);
            ctxs.push(DynamicContext::Mat(MatContext::ReLU {
                input: input_for_ctx,
            }));
        } else if let Some(_) = layer.as_sigmoid() {
            current = gpu_compute.run_sigmoid_forward(&current);
            ctxs.push(DynamicContext::Mat(MatContext::Sigmoid {
                output: current.clone(),
            }));
        } else if let Some(_) = layer.as_tanh() {
            current = gpu_compute.run_tanh_forward(&current);
            ctxs.push(DynamicContext::Mat(MatContext::Tanh {
                output: current.clone(),
            }));
        } else if let Some(leaky) = layer.as_leaky_relu() {
            let input_for_ctx = current.clone();
            current = gpu_compute.run_leaky_relu_forward(&current, leaky.alpha);
            ctxs.push(DynamicContext::Mat(MatContext::LeakyReLU {
                input: input_for_ctx,
            }));
        } else if layer.as_identity().is_some() {
            let (out, ctx) = layer.forward_mat(&current, params, slice);
            current = out;
            ctxs.push(ctx);
        } else {
            let (out, ctx) = layer.forward_mat(&current, params, slice);
            current = out;
            ctxs.push(ctx);
        }
    }

    (current, ctxs)
}

/// Обратный проход на GPU с persistent-буферами (старая версия).
/// Принимает `Mat` градиент выхода, возвращает `Mat` градиент входа.
pub fn process_backward_gpu(
    gpu_compute: &GpuCompute,
    _segment_buffers: &SegmentPersistentBuffers,
    layers: &[Box<dyn UniversalLayer>],
    slices: &[ParamSlice],
    contexts: &[DynamicContext],
    params: &[f32],
    grad_output: &Mat<f32>,
    total_grad: &mut [f32],
) -> Mat<f32> {
    let num_layers = layers.len();
    assert_eq!(contexts.len(), num_layers);
    assert_eq!(slices.len(), num_layers);

    let mut current_grad = grad_output.clone();

    for idx in (0..num_layers).rev() {
        let layer = &layers[idx];
        let slice = &slices[idx];
        let ctx = &contexts[idx];

        if let Some(linear) = layer.as_linear() {
            let (weight, _) = linear.get_weight_matrix_and_bias(params, slice);
            let input_mat = match ctx {
                DynamicContext::Mat(MatContext::Linear { input }) => input.clone(),
                _ => panic!("Expected Linear context"),
            };
            let (dx, dw, db) = gpu_compute.run_linear_backward(&input_mat, &weight, &current_grad);
            current_grad = dx;

            let in_feat = linear.input_features();
            let out_feat = linear.output_features();
            let w_start = slice.start;
            for r in 0..out_feat {
                for c in 0..in_feat {
                    total_grad[w_start + r * in_feat + c] += dw[(r, c)];
                }
            }
            let b_start = w_start + in_feat * out_feat;
            for (i, &v) in db.iter().enumerate() {
                total_grad[b_start + i] += v;
            }
        } else if let Some(_) = layer.as_relu() {
            let input_mat = match ctx {
                DynamicContext::Mat(MatContext::ReLU { input }) => input.clone(),
                _ => panic!("Expected ReLU context"),
            };
            current_grad = gpu_compute.run_relu_backward(&input_mat, &current_grad);
        } else if let Some(_) = layer.as_sigmoid() {
            let output_mat = match ctx {
                DynamicContext::Mat(MatContext::Sigmoid { output }) => output.clone(),
                _ => panic!("Expected Sigmoid context"),
            };
            current_grad = gpu_compute.run_sigmoid_backward(&output_mat, &current_grad);
        } else if let Some(_) = layer.as_tanh() {
            let output_mat = match ctx {
                DynamicContext::Mat(MatContext::Tanh { output }) => output.clone(),
                _ => panic!("Expected Tanh context"),
            };
            current_grad = gpu_compute.run_tanh_backward(&output_mat, &current_grad);
        } else if let Some(leaky) = layer.as_leaky_relu() {
            let input_mat = match ctx {
                DynamicContext::Mat(MatContext::LeakyReLU { input }) => input.clone(),
                _ => panic!("Expected LeakyReLU context"),
            };
            current_grad = gpu_compute.run_leaky_relu_backward(&input_mat, &current_grad, leaky.alpha);
        } else if let Some(_) = layer.as_softmax() {
            let output_mat = match ctx {
                DynamicContext::Mat(MatContext::Softmax { output }) => output.clone(),
                _ => panic!("Expected Softmax context"),
            };
            current_grad = gpu_compute.run_softmax_backward(&output_mat, &current_grad);
        } else if let Some(memory) = layer.as_memory() {
            current_grad = gpu_compute.run_memory_backward(&current_grad, memory.alpha);
        } else if let Some(softsparse) = layer.as_soft_sparse_gate() {
            let input_mat = match ctx {
                DynamicContext::Mat(MatContext::SoftSparseGate { input }) => input.clone(),
                _ => panic!("Expected SoftSparseGate context"),
            };
            let thresholds = &params[slice.start..slice.start + softsparse.in_features];
            let (dx, d_thr) = gpu_compute.run_softsparse_backward(
                &input_mat,
                &current_grad,
                thresholds,
                softsparse.temperature,
            );
            current_grad = dx;
            for (i, &g) in d_thr.iter().enumerate() {
                total_grad[slice.start + i] += g;
            }
        } else if let Some(softkeep) = layer.as_soft_keep_gate() {
            let input_mat = match ctx {
                DynamicContext::Mat(MatContext::SoftKeepGate { input }) => input.clone(),
                _ => panic!("Expected SoftKeepGate context"),
            };
            let thresholds = &params[slice.start..slice.start + softkeep.in_features];
            let (dx, d_thr) = gpu_compute.run_softkeep_backward(
                &input_mat,
                &current_grad,
                thresholds,
                softkeep.temperature,
            );
            current_grad = dx;
            for (i, &g) in d_thr.iter().enumerate() {
                total_grad[slice.start + i] += g;
            }
        } else if let Some(dual) = layer.as_dual_anchor() {
            let input_mat = match ctx {
                DynamicContext::Mat(MatContext::DualAnchor1D { input }) => input.clone(),
                _ => panic!("Expected DualAnchor1D context"),
            };
            let features = dual.features;
            let min_vals = &params[slice.start..slice.start + features];
            let max_vals = &params[slice.start + features..slice.start + 2 * features];
            let alpha = params[slice.start + 2 * features];
            let (dx, grad) = gpu_compute.run_dualanchor_backward(
                &input_mat,
                &current_grad,
                min_vals,
                max_vals,
                alpha,
            );
            current_grad = dx;
            for (i, &g) in grad.iter().enumerate() {
                total_grad[slice.start + i] += g;
            }
        } else {
            let (dx, layer_grad) = layer.backward_mat(ctx, &current_grad, params, slice);
            current_grad = dx;
            for (i, &g) in layer_grad.iter().enumerate() {
                total_grad[slice.start + i] += g;
            }
        }
    }

    current_grad
}

// ===================================================================
// НОВЫЕ БУФЕРИЗОВАННЫЕ ВЕРСИИ ДЛЯ GPU (MatrixBuffer)
// ===================================================================

/// Прямой проход на GPU с использованием MatrixBuffer (GPU-буферы остаются в VRAM).
/// Вход и выход — GPU MatrixBuffer. Контексты временно создаются как CPU-копии
/// (для совместимости с текущим DynamicContext).
pub fn process_forward_gpu_buffered(
    gpu_compute: &GpuCompute,
    _segment_buffers: &SegmentPersistentBuffers,
    layers: &[Box<dyn UniversalLayer>],
    slices: &[ParamSlice],
    params: &[f32],
    input: MatrixBuffer,
) -> (MatrixBuffer, Vec<DynamicContext>) {
    assert!(input.is_gpu(), "Input must be GPU buffer");
    let mut current = input;
    let mut ctxs = Vec::with_capacity(layers.len());

    for (layer, slice) in layers.iter().zip(slices.iter()) {
        if let Some(linear) = layer.as_linear() {
            // Временный fallback на CPU для Linear (будет оптимизировано позже)
            let input_mat = gpu_compute.download_gpu_matrix_to_mat(&current);
            let (weight, bias) = linear.get_weight_matrix_and_bias(params, slice);
            let (out_mat, ctx) = linear.forward_mat(&input_mat, params, slice);
            current = gpu_compute.upload_mat_to_gpu_matrix(&out_mat);
            ctxs.push(ctx); // ctx содержит input: input_mat
        } else if let Some(_) = layer.as_relu() {
            let input_mat = gpu_compute.download_gpu_matrix_to_mat(&current);
            current = gpu_compute.run_relu_forward_buffered(&current);
            ctxs.push(DynamicContext::Mat(MatContext::ReLU { input: input_mat }));
        } else if let Some(_) = layer.as_sigmoid() {
            current = gpu_compute.run_sigmoid_forward_buffered(&current);
            let output_mat = gpu_compute.download_gpu_matrix_to_mat(&current);
            ctxs.push(DynamicContext::Mat(MatContext::Sigmoid { output: output_mat }));
        } else if let Some(_) = layer.as_tanh() {
            current = gpu_compute.run_tanh_forward_buffered(&current);
            let output_mat = gpu_compute.download_gpu_matrix_to_mat(&current);
            ctxs.push(DynamicContext::Mat(MatContext::Tanh { output: output_mat }));
        } else if let Some(leaky) = layer.as_leaky_relu() {
            let input_mat = gpu_compute.download_gpu_matrix_to_mat(&current);
            current = gpu_compute.run_leaky_relu_forward_buffered(&current, leaky.alpha);
            ctxs.push(DynamicContext::Mat(MatContext::LeakyReLU { input: input_mat }));
        } else if let Some(_) = layer.as_softmax() {
            current = gpu_compute.run_softmax_forward_buffered(&current);
            let output_mat = gpu_compute.download_gpu_matrix_to_mat(&current);
            ctxs.push(DynamicContext::Mat(MatContext::Softmax { output: output_mat }));
        } else if let Some(_) = layer.as_identity() {
            // Identity: оставляем current без изменений, сохраняем контекст с входом (CPU-копия)
            let input_mat = gpu_compute.download_gpu_matrix_to_mat(&current);
            // current не изменяется
            ctxs.push(DynamicContext::Mat(MatContext::Identity { input: input_mat }));
        } else if let Some(memory) = layer.as_memory() {
            let input_mat = gpu_compute.download_gpu_matrix_to_mat(&current);
            current = gpu_compute.run_memory_forward_buffered(&current, memory.alpha);
            ctxs.push(DynamicContext::Mat(MatContext::Memory { input: input_mat }));
        } else if let Some(softsparse) = layer.as_soft_sparse_gate() {
            let input_mat = gpu_compute.download_gpu_matrix_to_mat(&current);
            let thresholds = &params[slice.start..slice.start + softsparse.in_features];
            current = gpu_compute.run_softsparse_forward_buffered(&current, thresholds, softsparse.temperature);
            ctxs.push(DynamicContext::Mat(MatContext::SoftSparseGate { input: input_mat }));
        } else if let Some(softkeep) = layer.as_soft_keep_gate() {
            let input_mat = gpu_compute.download_gpu_matrix_to_mat(&current);
            let thresholds = &params[slice.start..slice.start + softkeep.in_features];
            current = gpu_compute.run_softkeep_forward_buffered(&current, thresholds, softkeep.temperature);
            ctxs.push(DynamicContext::Mat(MatContext::SoftKeepGate { input: input_mat }));
        } else if let Some(dual) = layer.as_dual_anchor() {
            let input_mat = gpu_compute.download_gpu_matrix_to_mat(&current);
            let features = dual.features;
            let min_vals = &params[slice.start..slice.start + features];
            let max_vals = &params[slice.start + features..slice.start + 2 * features];
            let alpha = params[slice.start + 2 * features];
            current = gpu_compute.run_dualanchor_forward_buffered(&current, min_vals, max_vals, alpha);
            ctxs.push(DynamicContext::Mat(MatContext::DualAnchor1D { input: input_mat }));
        } else {
            // Fallback: CPU вычисления с загрузкой/выгрузкой
            let input_mat = gpu_compute.download_gpu_matrix_to_mat(&current);
            let (out_mat, ctx) = layer.forward_mat(&input_mat, params, slice);
            current = gpu_compute.upload_mat_to_gpu_matrix(&out_mat);
            ctxs.push(ctx);
        }
    }

    (current, ctxs)
}

/// Обратный проход на GPU с использованием MatrixBuffer (GPU-буферы остаются в VRAM).
/// Входной градиент — GPU MatrixBuffer, выходной градиент — GPU MatrixBuffer.
pub fn process_backward_gpu_buffered(
    gpu_compute: &GpuCompute,
    _segment_buffers: &SegmentPersistentBuffers,
    layers: &[Box<dyn UniversalLayer>],
    slices: &[ParamSlice],
    contexts: &[DynamicContext],
    params: &[f32],
    grad_output: MatrixBuffer,
    total_grad: &mut [f32],
) -> MatrixBuffer {
    assert!(grad_output.is_gpu(), "Grad output must be GPU");
    let num_layers = layers.len();
    assert_eq!(contexts.len(), num_layers);
    assert_eq!(slices.len(), num_layers);

    let mut current_grad = grad_output;

    for idx in (0..num_layers).rev() {
        let layer = &layers[idx];
        let slice = &slices[idx];
        let ctx = &contexts[idx];

        if let Some(linear) = layer.as_linear() {
            // Fallback на CPU для Linear
            let input_mat = match ctx {
                DynamicContext::Mat(MatContext::Linear { input }) => input.clone(),
                _ => panic!("Expected Linear context"),
            };
            let (weight, _) = linear.get_weight_matrix_and_bias(params, slice);
            let grad_out_mat = gpu_compute.download_gpu_matrix_to_mat(&current_grad);
            let (dx_mat, dw, db) = gpu_compute.run_linear_backward(&input_mat, &weight, &grad_out_mat);
            current_grad = gpu_compute.upload_mat_to_gpu_matrix(&dx_mat);

            let in_feat = linear.input_features();
            let out_feat = linear.output_features();
            let w_start = slice.start;
            for r in 0..out_feat {
                for c in 0..in_feat {
                    total_grad[w_start + r * in_feat + c] += dw[(r, c)];
                }
            }
            let b_start = w_start + in_feat * out_feat;
            for (i, &v) in db.iter().enumerate() {
                total_grad[b_start + i] += v;
            }
        } else if let Some(_) = layer.as_relu() {
            let input_mat = match ctx {
                DynamicContext::Mat(MatContext::ReLU { input }) => input.clone(),
                _ => panic!("Expected ReLU context"),
            };
            // Для ReLU входной буфер нужен GPU; у нас есть только CPU-копия.
            // Временно загружаем её на GPU.
            let input_gpu = gpu_compute.upload_mat_to_gpu_matrix(&input_mat);
            current_grad = gpu_compute.run_relu_backward_buffered(&input_gpu, &current_grad);
        } else if let Some(_) = layer.as_sigmoid() {
            let output_mat = match ctx {
                DynamicContext::Mat(MatContext::Sigmoid { output }) => output.clone(),
                _ => panic!("Expected Sigmoid context"),
            };
            let output_gpu = gpu_compute.upload_mat_to_gpu_matrix(&output_mat);
            current_grad = gpu_compute.run_sigmoid_backward_buffered(&output_gpu, &current_grad);
        } else if let Some(_) = layer.as_tanh() {
            let output_mat = match ctx {
                DynamicContext::Mat(MatContext::Tanh { output }) => output.clone(),
                _ => panic!("Expected Tanh context"),
            };
            let output_gpu = gpu_compute.upload_mat_to_gpu_matrix(&output_mat);
            current_grad = gpu_compute.run_tanh_backward_buffered(&output_gpu, &current_grad);
        } else if let Some(leaky) = layer.as_leaky_relu() {
            let input_mat = match ctx {
                DynamicContext::Mat(MatContext::LeakyReLU { input }) => input.clone(),
                _ => panic!("Expected LeakyReLU context"),
            };
            let input_gpu = gpu_compute.upload_mat_to_gpu_matrix(&input_mat);
            current_grad = gpu_compute.run_leaky_relu_backward_buffered(&input_gpu, &current_grad, leaky.alpha);
        } else if let Some(_) = layer.as_softmax() {
            let output_mat = match ctx {
                DynamicContext::Mat(MatContext::Softmax { output }) => output.clone(),
                _ => panic!("Expected Softmax context"),
            };
            let output_gpu = gpu_compute.upload_mat_to_gpu_matrix(&output_mat);
            current_grad = gpu_compute.run_softmax_backward_buffered(&output_gpu, &current_grad);
        } else if let Some(memory) = layer.as_memory() {
            current_grad = gpu_compute.run_memory_backward_buffered(&current_grad, memory.alpha);
        } else if let Some(softsparse) = layer.as_soft_sparse_gate() {
            let input_mat = match ctx {
                DynamicContext::Mat(MatContext::SoftSparseGate { input }) => input.clone(),
                _ => panic!("Expected SoftSparseGate context"),
            };
            let input_gpu = gpu_compute.upload_mat_to_gpu_matrix(&input_mat);
            let thresholds = &params[slice.start..slice.start + softsparse.in_features];
            let (dx, d_thr) = gpu_compute.run_softsparse_backward_buffered(
                &input_gpu,
                &current_grad,
                thresholds,
                softsparse.temperature,
            );
            current_grad = dx;
            for (i, &g) in d_thr.iter().enumerate() {
                total_grad[slice.start + i] += g;
            }
        } else if let Some(softkeep) = layer.as_soft_keep_gate() {
            let input_mat = match ctx {
                DynamicContext::Mat(MatContext::SoftKeepGate { input }) => input.clone(),
                _ => panic!("Expected SoftKeepGate context"),
            };
            let input_gpu = gpu_compute.upload_mat_to_gpu_matrix(&input_mat);
            let thresholds = &params[slice.start..slice.start + softkeep.in_features];
            let (dx, d_thr) = gpu_compute.run_softkeep_backward_buffered(
                &input_gpu,
                &current_grad,
                thresholds,
                softkeep.temperature,
            );
            current_grad = dx;
            for (i, &g) in d_thr.iter().enumerate() {
                total_grad[slice.start + i] += g;
            }
        } else if let Some(dual) = layer.as_dual_anchor() {
            let input_mat = match ctx {
                DynamicContext::Mat(MatContext::DualAnchor1D { input }) => input.clone(),
                _ => panic!("Expected DualAnchor1D context"),
            };
            let input_gpu = gpu_compute.upload_mat_to_gpu_matrix(&input_mat);
            let features = dual.features;
            let min_vals = &params[slice.start..slice.start + features];
            let max_vals = &params[slice.start + features..slice.start + 2 * features];
            let alpha = params[slice.start + 2 * features];
            let (dx, grad) = gpu_compute.run_dualanchor_backward_buffered(
                &input_gpu,
                &current_grad,
                min_vals,
                max_vals,
                alpha,
            );
            current_grad = dx;
            for (i, &g) in grad.iter().enumerate() {
                total_grad[slice.start + i] += g;
            }
        } else {
            // Fallback: CPU вычисления
            let grad_out_mat = gpu_compute.download_gpu_matrix_to_mat(&current_grad);
            let (dx_mat, layer_grad) = layer.backward_mat(ctx, &grad_out_mat, params, slice);
            current_grad = gpu_compute.upload_mat_to_gpu_matrix(&dx_mat);
            for (i, &g) in layer_grad.iter().enumerate() {
                total_grad[slice.start + i] += g;
            }
        }
    }

    current_grad
}