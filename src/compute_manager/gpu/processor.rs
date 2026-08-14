// src/compute_manager/gpu/processor.rs

use faer::Mat;

use crate::compute_manager::graph::types::DynamicContext;
use crate::compute_manager::persistent_buffer::SegmentPersistentBuffers;
use crate::compute_manager::matrix_buffer::MatrixBufferHandle;
use crate::layers::buffered_context::BufferedContext;
use crate::layers::mat_context::MatContext;
use crate::layers::UniversalLayer;
use crate::model_plan::param_store::ParamSlice;

use super::compute::GpuCompute;

// Вспомогательные функции для конвертации между Mat и GPU-дескрипторами

/// Создаёт GPU-дескриптор из `Mat<f32>`.
fn upload_mat_to_gpu_handle(gpu: &GpuCompute, mat: &Mat<f32>) -> MatrixBufferHandle {
    let flat = GpuCompute::mat_to_flat(mat);
    gpu.upload_vec_to_gpu_handle(&flat, mat.nrows(), mat.ncols())
}

/// Скачивает GPU-дескриптор в `Mat<f32>`.
fn download_gpu_handle_to_mat(gpu: &GpuCompute, handle: &MatrixBufferHandle) -> Mat<f32> {
    let vec = gpu.download_gpu_handle_to_vec(handle);
    Mat::from_fn(handle.rows(), handle.cols(), |r, c| {
        vec[c * handle.rows() + r]
    })
}

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
// НОВЫЕ БУФЕРИЗОВАННЫЕ ВЕРСИИ ДЛЯ GPU (MatrixBufferHandle)
// ===================================================================

/// Прямой проход на GPU с использованием MatrixBufferHandle.
/// Вход и выход — GPU-дескрипторы. Контексты создаются как Buffered.
///
/// Примечание: для слоя Memory требуется предварительная инициализация
/// внутреннего состояния GPU через `GpuCompute::init_memory_state`.
/// В текущей реализации эта инициализация выполняется в вызывающем коде
/// до вызова данной функции (см. `MixedModel::forward_mat_multi_buffered`).
pub fn process_forward_gpu_buffered(
    gpu_compute: &GpuCompute,
    _segment_buffers: &SegmentPersistentBuffers,
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
            // Identity: копируем данные GPU -> GPU
            let out_handle = gpu_compute.allocate_gpu_matrix_handle(current.rows(), current.cols());
            gpu_compute.copy_gpu_handle_to_gpu_handle(&current, &out_handle);
            ctxs.push(DynamicContext::Buffered(BufferedContext::Identity {
                input: current.clone(),
            }));
            current = out_handle;
        } else if let Some(memory) = layer.as_memory() {
            // Инициализируем состояние Memory, если ещё не инициализировано.
            // Внимание: `memory_state` должен быть инициализирован заранее;
            // в противном случае здесь произойдёт паника при вызове handle-метода.
            // В вызывающем коде должна быть выполнена инициализация.
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
            // Неизвестный слой – такого не должно быть, так как все слои перечислены.
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
/// Для всех слоёв используются GPU handle-методы.
pub fn process_backward_gpu_buffered(
    gpu_compute: &GpuCompute,
    _segment_buffers: &SegmentPersistentBuffers,
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

            // Извлекаем входной handle из контекста
            let input_handle = match ctx {
                DynamicContext::Buffered(BufferedContext::Linear { input }) => input.clone(),
                DynamicContext::Mat(MatContext::Linear { input }) => {
                    // Конвертируем Mat в GPU handle (на случай, если контекст старый)
                    upload_mat_to_gpu_handle(gpu_compute, input)
                }
                _ => panic!("Expected Linear context"),
            };

            // Веса как GPU handle
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

            // Записываем градиенты в total_grad
            let grad_weight_vec = gpu_compute.download_gpu_handle_to_vec(&grad_weight_handle);
            for (i, &g) in grad_weight_vec.iter().enumerate() {
                total_grad[w_start + i] += g;
            }
            for (i, &g) in grad_bias.iter().enumerate() {
                total_grad[b_start + i] += g;
            }

            current_grad = grad_input_handle;
        } else if let Some(_) = layer.as_relu() {
            let input_handle = match ctx {
                DynamicContext::Buffered(BufferedContext::ReLU { input }) => input.clone(),
                DynamicContext::Mat(MatContext::ReLU { input }) => {
                    upload_mat_to_gpu_handle(gpu_compute, input)
                }
                _ => panic!("Expected ReLU context"),
            };
            let grad_input_handle = gpu_compute.allocate_gpu_matrix_handle(current_grad.rows(), current_grad.cols());
            gpu_compute.run_relu_backward_buffered_handle(
                &input_handle,
                &current_grad,
                &grad_input_handle,
            );
            current_grad = grad_input_handle;
        } else if let Some(_) = layer.as_sigmoid() {
            let output_handle = match ctx {
                DynamicContext::Buffered(BufferedContext::Sigmoid { output }) => output.clone(),
                DynamicContext::Mat(MatContext::Sigmoid { output }) => {
                    upload_mat_to_gpu_handle(gpu_compute, output)
                }
                _ => panic!("Expected Sigmoid context"),
            };
            let grad_input_handle = gpu_compute.allocate_gpu_matrix_handle(current_grad.rows(), current_grad.cols());
            gpu_compute.run_sigmoid_backward_buffered_handle(
                &output_handle,
                &current_grad,
                &grad_input_handle,
            );
            current_grad = grad_input_handle;
        } else if let Some(_) = layer.as_tanh() {
            let output_handle = match ctx {
                DynamicContext::Buffered(BufferedContext::Tanh { output }) => output.clone(),
                DynamicContext::Mat(MatContext::Tanh { output }) => {
                    upload_mat_to_gpu_handle(gpu_compute, output)
                }
                _ => panic!("Expected Tanh context"),
            };
            let grad_input_handle = gpu_compute.allocate_gpu_matrix_handle(current_grad.rows(), current_grad.cols());
            gpu_compute.run_tanh_backward_buffered_handle(
                &output_handle,
                &current_grad,
                &grad_input_handle,
            );
            current_grad = grad_input_handle;
        } else if let Some(leaky) = layer.as_leaky_relu() {
            let input_handle = match ctx {
                DynamicContext::Buffered(BufferedContext::LeakyReLU { input }) => input.clone(),
                DynamicContext::Mat(MatContext::LeakyReLU { input }) => {
                    upload_mat_to_gpu_handle(gpu_compute, input)
                }
                _ => panic!("Expected LeakyReLU context"),
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
            let output_handle = match ctx {
                DynamicContext::Buffered(BufferedContext::Softmax { output }) => output.clone(),
                DynamicContext::Mat(MatContext::Softmax { output }) => {
                    upload_mat_to_gpu_handle(gpu_compute, output)
                }
                _ => panic!("Expected Softmax context"),
            };
            let grad_input_handle = gpu_compute.allocate_gpu_matrix_handle(current_grad.rows(), current_grad.cols());
            gpu_compute.run_softmax_backward_buffered_handle(
                &output_handle,
                &current_grad,
                &grad_input_handle,
            );
            current_grad = grad_input_handle;
        } else if let Some(_) = layer.as_identity() {
            // Identity: градиент проходит насквозь, копируем GPU -> GPU
            let grad_input_handle = gpu_compute.allocate_gpu_matrix_handle(current_grad.rows(), current_grad.cols());
            gpu_compute.copy_gpu_handle_to_gpu_handle(&current_grad, &grad_input_handle);
            current_grad = grad_input_handle;
        } else if let Some(memory) = layer.as_memory() {
            // Memory обратный проход: gi = go * (1 - alpha)
            let grad_input_handle = gpu_compute.allocate_gpu_matrix_handle(current_grad.rows(), current_grad.cols());
            gpu_compute.run_memory_backward_buffered_handle(
                &current_grad,
                &grad_input_handle,
                memory.alpha,
            );
            current_grad = grad_input_handle;
        } else if let Some(soft_sparse) = layer.as_soft_sparse_gate() {
            let input_handle = match ctx {
                DynamicContext::Buffered(BufferedContext::SoftSparseGate { input }) => input.clone(),
                DynamicContext::Mat(MatContext::SoftSparseGate { input }) => {
                    upload_mat_to_gpu_handle(gpu_compute, input)
                }
                _ => panic!("Expected SoftSparseGate context"),
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
            let input_handle = match ctx {
                DynamicContext::Buffered(BufferedContext::SoftKeepGate { input }) => input.clone(),
                DynamicContext::Mat(MatContext::SoftKeepGate { input }) => {
                    upload_mat_to_gpu_handle(gpu_compute, input)
                }
                _ => panic!("Expected SoftKeepGate context"),
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
            let input_handle = match ctx {
                DynamicContext::Buffered(BufferedContext::DualAnchor1D { input }) => input.clone(),
                DynamicContext::Mat(MatContext::DualAnchor1D { input }) => {
                    upload_mat_to_gpu_handle(gpu_compute, input)
                }
                _ => panic!("Expected DualAnchor1D context"),
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