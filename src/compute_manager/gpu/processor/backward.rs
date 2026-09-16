// src/compute_manager/gpu/processor/backward.rs

use crate::compute_manager::graph::types::DynamicContext;
use crate::compute_manager::gpu::compute::GpuCompute;
use crate::compute_manager::matrix_buffer::MatrixBufferHandle;
use crate::layers::UniversalLayer;
use crate::model_plan::param_store::ParamSlice;

use super::layers;

/// Обратный проход на GPU с использованием MatrixBufferHandle.
/// Входной градиент — GPU-дескриптор, выходной градиент — GPU-дескриптор.
/// Градиенты параметров записываются напрямую в `grad_params_handle` (GPU).
pub fn process_backward_gpu_buffered(
    gpu_compute: &GpuCompute,
    layers_list: &[Box<dyn UniversalLayer>],
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

    // ========================================================================
    // КРИТИЧНО: обнуляем буфер градиентов параметров перед backward.
    //
    // GPU-шейдеры слоёв накапливают градиенты по параметрам через
    // ATOMIC_ADD_FLOAT (CAS-loop). Без обнуления атомарные сложения будут
    // добавляться к градиентам от ПРЕДЫДУЩЕГО backward, что приводит к
    // экспоненциальному накоплению. CPU-ветка от этой проблемы свободна:
    // CPU-реализации пишут градиенты параметров напрямую (gp[i] = sum).
    // ========================================================================
    if grad_params_handle.is_gpu()
        && grad_params_handle.rows() * grad_params_handle.cols() > 0
    {
        gpu_compute.fill_gpu_handle(grad_params_handle, 0.0);
    }

    let num_layers = layers_list.len();
    assert_eq!(contexts.len(), num_layers);
    assert_eq!(slices.len(), num_layers);

    let mut current_grad = grad_output;

    for idx in (0..num_layers).rev() {
        let layer = &layers_list[idx];
        let slice = &slices[idx];
        let ctx = &contexts[idx];
        let layer_ref: &dyn UniversalLayer = layer.as_ref();

        let new_grad = if let Some(r) = layers::linear::backward(
            gpu_compute,
            layer_ref,
            ctx,
            &current_grad,
            params_handle,
            slice,
            grad_params_handle,
        ) {
            r
        } else if let Some(r) = layers::activation::backward(
            gpu_compute,
            layer_ref,
            ctx,
            &current_grad,
            params_handle,
            slice,
            grad_params_handle,
        ) {
            r
        } else if let Some(r) = layers::memory::backward(
            gpu_compute,
            layer_ref,
            ctx,
            &current_grad,
            params_handle,
            slice,
            grad_params_handle,
        ) {
            r
        } else if let Some(r) = layers::gate::backward(
            gpu_compute,
            layer_ref,
            ctx,
            &current_grad,
            params_handle,
            slice,
            grad_params_handle,
        ) {
            r
        } else if let Some(r) = layers::anchor::backward(
            gpu_compute,
            layer_ref,
            ctx,
            &current_grad,
            params_handle,
            slice,
            grad_params_handle,
        ) {
            r
        } else if let Some(r) = layers::norm::backward(
            gpu_compute,
            layer_ref,
            ctx,
            &current_grad,
            params_handle,
            slice,
            grad_params_handle,
        ) {
            r
        } else if let Some(r) = layers::dropout::backward(
            gpu_compute,
            layer_ref,
            ctx,
            &current_grad,
            params_handle,
            slice,
            grad_params_handle,
        ) {
            r
        } else if let Some(r) = layers::kan::backward(
            gpu_compute,
            layer_ref,
            ctx,
            &current_grad,
            params_handle,
            slice,
            grad_params_handle,
        ) {
            r
        } else if let Some(r) = layers::feature_fusion::backward(
            gpu_compute,
            layer_ref,
            ctx,
            &current_grad,
            params_handle,
            slice,
            grad_params_handle,
        ) {
            r
        } else if let Some(r) = layers::spectral_norm::backward(
            gpu_compute,
            layer_ref,
            ctx,
            &current_grad,
            params_handle,
            slice,
            grad_params_handle,
        ) {
            r
        } else if let Some(r) = layers::attention::backward(
            gpu_compute,
            layer_ref,
            ctx,
            &current_grad,
            params_handle,
            slice,
            grad_params_handle,
        ) {
            r
        } else if let Some(r) = layers::recurrent::backward(
            gpu_compute,
            layer_ref,
            ctx,
            &current_grad,
            params_handle,
            slice,
            grad_params_handle,
        ) {
            r
        } else {
            panic!(
                "Unsupported layer in GPU buffered backward: {:?}",
                std::any::type_name_of_val(layer.as_ref())
            );
        };

        current_grad = new_grad;
    }

    current_grad
}