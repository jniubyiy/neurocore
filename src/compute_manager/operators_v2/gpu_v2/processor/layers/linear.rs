// src/compute_manager/gpu/processor/layers/linear.rs

use crate::compute_manager::core::dynamic_context::DynamicContext;
use crate::compute_manager::operators_v2::gpu_v2::compute::GpuCompute;
use crate::compute_manager::operators_v2::memory_v2::buffer::view::MatrixBufferView;
use crate::compute_manager::operators_v2::memory_v2::buffer::MatrixBufferHandle;
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
    let linear = layer.as_linear()?;
    let in_feat = linear.input_features();
    let out_feat = linear.output_features();
    let w_start = slice.start;
    let b_start = w_start + in_feat * out_feat;

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

    let out_handle = gpu.allocate_gpu_matrix_handle(input.rows(), out_feat);
    gpu.run_linear_forward_buffered_handle(input, &weight_view, &bias_view, &out_handle);

    let ctx = DynamicContext::Buffered(BufferedContext::Linear {
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
    let linear = layer.as_linear()?;
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

    let grad_input_handle =
        gpu.allocate_gpu_matrix_handle(grad_output.rows(), in_feat);
    gpu.run_linear_backward_buffered_handle(
        &input_handle,
        &weight_view,
        grad_output,
        &grad_input_handle,
        &grad_weight_view,
        &grad_bias_view,
    );
    Some(grad_input_handle)
}