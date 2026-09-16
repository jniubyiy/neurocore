// src/compute_manager/gpu/processor/layers/gate.rs
//
// Категория «гейты»:
//   SoftSparseGate, SoftKeepGate, SparseFeatureSelectionGate.

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
    // ---- SoftSparseGate ----
    if let Some(soft_sparse) = layer.as_soft_sparse_gate() {
        let features = soft_sparse.in_features;
        let thresholds_view =
            MatrixBufferView::new(params_handle.clone(), slice.start, features);
        let out_handle = gpu.allocate_gpu_matrix_handle(input.rows(), input.cols());
        gpu.run_softsparse_forward_buffered_handle(
            input,
            &thresholds_view,
            soft_sparse.temperature,
            &out_handle,
        );
        let ctx = DynamicContext::Buffered(BufferedContext::SoftSparseGate {
            input: input.clone(),
        });
        return Some((out_handle, ctx));
    }

    // ---- SoftKeepGate ----
    if let Some(soft_keep) = layer.as_soft_keep_gate() {
        let features = soft_keep.in_features;
        let thresholds_view =
            MatrixBufferView::new(params_handle.clone(), slice.start, features);
        let out_handle = gpu.allocate_gpu_matrix_handle(input.rows(), input.cols());
        gpu.run_softkeep_forward_buffered_handle(
            input,
            &thresholds_view,
            soft_keep.temperature,
            &out_handle,
        );
        let ctx = DynamicContext::Buffered(BufferedContext::SoftKeepGate {
            input: input.clone(),
        });
        return Some((out_handle, ctx));
    }

    // ---- SparseFeatureSelectionGate ----
    if let Some(sfs) = layer.as_sparse_feature_selection_gate() {
        let features = sfs.features;
        let params_len = features + 1;
        let params_view =
            MatrixBufferView::new(params_handle.clone(), slice.start, params_len);
        let out_handle = gpu.allocate_gpu_matrix_handle(input.rows(), features);
        gpu.run_sparse_feature_selection_forward_buffered_handle(
            input,
            &params_view,
            &out_handle,
        );
        let ctx = DynamicContext::Buffered(
            BufferedContext::SparseFeatureSelectionGate {
                input: input.clone(),
            },
        );
        return Some((out_handle, ctx));
    }

    None
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
    // ---- SoftSparseGate ----
    if let Some(soft_sparse) = layer.as_soft_sparse_gate() {
        let features = soft_sparse.in_features;
        let DynamicContext::Buffered(bc) = ctx;
        let input_handle = match bc {
            BufferedContext::SoftSparseGate { input } => input.clone(),
            _ => panic!("Expected SoftSparseGate Buffered context"),
        };
        let thresholds_view =
            MatrixBufferView::new(params_handle.clone(), slice.start, features);
        let grad_thresh_view =
            MatrixBufferView::new(grad_params_handle.clone(), slice.start, features);
        let gi =
            gpu.allocate_gpu_matrix_handle(grad_output.rows(), grad_output.cols());
        gpu.run_softsparse_backward_buffered_handle(
            &input_handle,
            grad_output,
            &thresholds_view,
            soft_sparse.temperature,
            &gi,
            &grad_thresh_view,
        );
        return Some(gi);
    }

    // ---- SoftKeepGate ----
    if let Some(soft_keep) = layer.as_soft_keep_gate() {
        let features = soft_keep.in_features;
        let DynamicContext::Buffered(bc) = ctx;
        let input_handle = match bc {
            BufferedContext::SoftKeepGate { input } => input.clone(),
            _ => panic!("Expected SoftKeepGate Buffered context"),
        };
        let thresholds_view =
            MatrixBufferView::new(params_handle.clone(), slice.start, features);
        let grad_thresh_view =
            MatrixBufferView::new(grad_params_handle.clone(), slice.start, features);
        let gi =
            gpu.allocate_gpu_matrix_handle(grad_output.rows(), grad_output.cols());
        gpu.run_softkeep_backward_buffered_handle(
            &input_handle,
            grad_output,
            &thresholds_view,
            soft_keep.temperature,
            &gi,
            &grad_thresh_view,
        );
        return Some(gi);
    }

    // ---- SparseFeatureSelectionGate ----
    if let Some(sfs) = layer.as_sparse_feature_selection_gate() {
        let features = sfs.features;
        let params_len = features + 1;
        let DynamicContext::Buffered(bc) = ctx;
        let input_handle = match bc {
            BufferedContext::SparseFeatureSelectionGate { input } => input.clone(),
            _ => panic!("Expected SparseFeatureSelectionGate Buffered context"),
        };
        let params_view =
            MatrixBufferView::new(params_handle.clone(), slice.start, params_len);
        let grad_params_view =
            MatrixBufferView::new(grad_params_handle.clone(), slice.start, params_len);
        let gi = gpu.allocate_gpu_matrix_handle(grad_output.rows(), features);
        gpu.run_sparse_feature_selection_backward_buffered_handle(
            &input_handle,
            grad_output,
            &params_view,
            &gi,
            &grad_params_view,
        );
        return Some(gi);
    }

    None
}