// src/compute_manager/gpu/processor/layers/kan.rs
//
// Категория «KAN»: MultiResolutionKANLinear.

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
    let kan = layer.as_multi_resolution_kan_linear()?;
    let in_features = kan.input_features();
    let out_features = kan.output_features();
    let params_len = kan.param_len();
    let params_view =
        MatrixBufferView::new(params_handle.clone(), slice.start, params_len);
    let out_handle = gpu.allocate_gpu_matrix_handle(input.rows(), out_features);
    gpu.run_multi_resolution_kan_linear_forward_buffered_handle(
        input,
        &params_view,
        &out_handle,
    );
    let ctx = DynamicContext::Buffered(
        BufferedContext::MultiResolutionKANLinear {
            input: input.clone(),
        },
    );
    let _ = in_features;
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
    let kan = layer.as_multi_resolution_kan_linear()?;
    let in_features = kan.input_features();
    let out_features = kan.output_features();
    let params_len = kan.param_len();
    let DynamicContext::Buffered(bc) = ctx;
    let input_handle = match bc {
        BufferedContext::MultiResolutionKANLinear { input } => input.clone(),
        _ => panic!("Expected MultiResolutionKANLinear Buffered context"),
    };
    let params_view =
        MatrixBufferView::new(params_handle.clone(), slice.start, params_len);
    let grad_params_view =
        MatrixBufferView::new(grad_params_handle.clone(), slice.start, params_len);
    let gi = gpu.allocate_gpu_matrix_handle(grad_output.rows(), in_features);
    gpu.run_multi_resolution_kan_linear_backward_buffered_handle(
        &input_handle,
        grad_output,
        &params_view,
        &gi,
        &grad_params_view,
    );
    let _ = out_features;
    Some(gi)
}