// src/compute_manager/gpu/processor/layers/memory.rs

use crate::compute_manager::core::dynamic_context::DynamicContext;
use crate::compute_manager::operators_v2::gpu_v2::compute::GpuCompute;
use crate::compute_manager::operators_v2::memory_v2::buffer::MatrixBufferHandle;
use crate::layers::buffered_context::BufferedContext;
use crate::layers::UniversalLayer;
use crate::model_plan::param_store::ParamSlice;

/// Прямой проход Memory на GPU.
///
/// В отличие от остальных обработчиков, дополнительно принимает `memory_idx` —
/// порядковый номер Memory-слоя в сегменте. Он используется как ключ в
/// `gpu_compute.memory_states` (per-slot состояние якорей `min_cells`/`max_cells`).
/// После обработки счётчик инкрементируется.
pub fn forward(
    gpu: &GpuCompute,
    layer: &dyn UniversalLayer,
    input: &MatrixBufferHandle,
    _params_handle: &MatrixBufferHandle,
    _slice: &ParamSlice,
    memory_idx: &mut usize,
) -> Option<(MatrixBufferHandle, DynamicContext)> {
    let memory = layer.as_memory()?;

    let out_handle = gpu.allocate_gpu_matrix_handle(input.rows(), input.cols());
    gpu.run_memory_forward_buffered_handle(
        input,
        &out_handle,
        memory.alpha,
        *memory_idx,
    );
    *memory_idx += 1;

    // GPU-путь хранит якоря Memory в gpu_compute.memory_states.
    // Поля min_cells/max_cells в контексте — заглушки для
    // согласованности с вариантом BufferedContext::Memory;
    // GPU-backward их не читает.
    let min_cells = gpu.allocate_gpu_matrix_handle(1, 1);
    let max_cells = gpu.allocate_gpu_matrix_handle(1, 1);

    let ctx = DynamicContext::Buffered(BufferedContext::Memory {
        input: input.clone(),
        min_cells,
        max_cells,
    });
    Some((out_handle, ctx))
}

pub fn backward(
    gpu: &GpuCompute,
    layer: &dyn UniversalLayer,
    _ctx: &DynamicContext,
    grad_output: &MatrixBufferHandle,
    _params_handle: &MatrixBufferHandle,
    _slice: &ParamSlice,
    _grad_params_handle: &MatrixBufferHandle,
) -> Option<MatrixBufferHandle> {
    let memory = layer.as_memory()?;

    let grad_input_handle =
        gpu.allocate_gpu_matrix_handle(grad_output.rows(), grad_output.cols());
    gpu.run_memory_backward_buffered_handle(
        grad_output,
        &grad_input_handle,
        memory.alpha,
    );
    Some(grad_input_handle)
}