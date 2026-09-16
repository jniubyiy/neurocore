// src/compute_manager/gpu/processor/layers/anchor.rs
//
// Категория «якорные слои»: DualAnchor.

use crate::compute_manager::graph::types::DynamicContext;
use crate::compute_manager::gpu::compute::GpuCompute;
use crate::compute_manager::matrix_buffer::view::MatrixBufferView;
use crate::compute_manager::matrix_buffer::MatrixBufferHandle;
use crate::layers::buffered_context::BufferedContext;
use crate::layers::UniversalLayer;
use crate::model_plan::param_store::ParamSlice;

pub fn forward(
    gpu: &GpuCompute,
    layer: &dyn UniversalLayer,
    input: &MatrixBufferHandle,
    params_handle: &MatrixBufferHandle,
    slice: &ParamSlice,
) -> Option<(MatrixBufferHandle, DynamicContext)> {
    let dual = layer.as_dual_anchor()?;
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
    let out_handle = gpu.allocate_gpu_matrix_handle(input.rows(), input.cols());
    gpu.run_dualanchor_forward_buffered_handle(
        input,
        &min_view,
        &max_view,
        &alpha_view,
        &out_handle,
    );
    let ctx = DynamicContext::Buffered(BufferedContext::DualAnchor1D {
        input: input.clone(),
    });
    Some((out_handle, ctx))
}

pub fn backward(
    gpu: &GpuCompute,
    layer: &dyn UniversalLayer,
    ctx: &DynamicContext,
    grad_output: &MatrixBufferHandle,
    params_handle: &MatrixBufferHandle,
    slice: &ParamSlice,
    grad_params_handle: &MatrixBufferHandle,
) -> Option<MatrixBufferHandle> {
    let dual = layer.as_dual_anchor()?;
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

    let grad_params_view = MatrixBufferView::new(
        grad_params_handle.clone(),
        slice.start,
        2 * features + 1,
    );

    let gi = gpu.allocate_gpu_matrix_handle(grad_output.rows(), grad_output.cols());
    gpu.run_dualanchor_backward_buffered_handle(
        &input_handle,
        grad_output,
        &min_view,
        &max_view,
        &alpha_view,
        &gi,
        &grad_params_view,
    );
    Some(gi)
}