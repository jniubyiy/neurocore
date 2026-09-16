// src/compute_manager/gpu/processor/forward.rs

use crate::compute_manager::graph::types::DynamicContext;
use crate::compute_manager::gpu::compute::GpuCompute;
use crate::compute_manager::matrix_buffer::MatrixBufferHandle;
use crate::layers::UniversalLayer;
use crate::model_plan::param_store::ParamSlice;

use super::layers;

/// Прямой проход на GPU с использованием MatrixBufferHandle.
/// Вход и выход — GPU-дескрипторы. Контексты создаются как Buffered.
///
/// Параметры сегмента уже должны находиться на GPU (в `params_handle`).
/// Доступ к отдельным слоям осуществляется через `MatrixBufferView`,
/// который представляет собой непрерывный диапазон внутри буфера.
///
/// # Per-chunk state
///
/// Слои, у которых есть состояние, передают свои state-буферы через
/// `BufferedContext`. На GPU большая часть состояния хранится в
/// собственных механизмах слоя (`gpu_compute.memory_states` для Memory,
/// глобальный `FORWARD_CACHE` для LinearAttention, временные буферы для
/// Mamba), но некоторые дескрипторы обязаны жить в контексте, потому что
/// CPU-backward и GPU-backward используют единый `BufferedContext`.
/// Где GPU-путь state не использует, в контекст кладутся заглушки.
pub fn process_forward_gpu_buffered(
    gpu_compute: &GpuCompute,
    layers_list: &[Box<dyn UniversalLayer>],
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
    let mut ctxs = Vec::with_capacity(layers_list.len());
    let mut memory_idx = 0usize;

    for (layer, slice) in layers_list.iter().zip(slices.iter()) {
        let layer_ref: &dyn UniversalLayer = layer.as_ref();

        let (out, ctx) = if let Some(r) = layers::linear::forward(
            gpu_compute,
            layer_ref,
            &current,
            params_handle,
            slice,
        ) {
            r
        } else if let Some(r) = layers::activation::forward(
            gpu_compute,
            layer_ref,
            &current,
            params_handle,
            slice,
        ) {
            r
        } else if let Some(r) = layers::memory::forward(
            gpu_compute,
            layer_ref,
            &current,
            params_handle,
            slice,
            &mut memory_idx,
        ) {
            r
        } else if let Some(r) = layers::gate::forward(
            gpu_compute,
            layer_ref,
            &current,
            params_handle,
            slice,
        ) {
            r
        } else if let Some(r) = layers::anchor::forward(
            gpu_compute,
            layer_ref,
            &current,
            params_handle,
            slice,
        ) {
            r
        } else if let Some(r) = layers::norm::forward(
            gpu_compute,
            layer_ref,
            &current,
            params_handle,
            slice,
        ) {
            r
        } else if let Some(r) = layers::dropout::forward(
            gpu_compute,
            layer_ref,
            &current,
            params_handle,
            slice,
        ) {
            r
        } else if let Some(r) = layers::kan::forward(
            gpu_compute,
            layer_ref,
            &current,
            params_handle,
            slice,
        ) {
            r
        } else if let Some(r) = layers::feature_fusion::forward(
            gpu_compute,
            layer_ref,
            &current,
            params_handle,
            slice,
        ) {
            r
        } else if let Some(r) = layers::spectral_norm::forward(
            gpu_compute,
            layer_ref,
            &current,
            params_handle,
            slice,
        ) {
            r
        } else if let Some(r) = layers::attention::forward(
            gpu_compute,
            layer_ref,
            &current,
            params_handle,
            slice,
        ) {
            r
        } else if let Some(r) = layers::recurrent::forward(
            gpu_compute,
            layer_ref,
            &current,
            params_handle,
            slice,
        ) {
            r
        } else {
            panic!(
                "Unsupported layer in GPU buffered forward: {:?}",
                std::any::type_name_of_val(layer.as_ref())
            );
        };

        ctxs.push(ctx);
        current = out;
    }

    (current, ctxs)
}