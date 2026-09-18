// src/compute_manager/cpu/parallel/dims.rs
//
// Определение размерностей слоёв для forward/backward.

use crate::compute_manager::operators_v2::memory_v2::buffer::MatrixBufferHandle;
use crate::layers::UniversalLayer;

/// Число выходных признаков слоя при заданном входном буфере.
#[inline]
pub(super) fn get_output_features(
    layer: &Box<dyn UniversalLayer>,
    input: &MatrixBufferHandle,
) -> usize {
    layer.output_features_for(input.cols())
}

/// Число входных признаков слоя при заданном буфере градиента.
#[inline]
pub(super) fn get_input_features(
    layer: &Box<dyn UniversalLayer>,
    grad_output: &MatrixBufferHandle,
) -> usize {
    layer.input_features_for(grad_output.cols())
}