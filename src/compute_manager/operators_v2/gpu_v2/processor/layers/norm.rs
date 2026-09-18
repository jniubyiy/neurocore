// src/compute_manager/gpu/processor/layers/norm.rs
//
// Категория «нормализации»:
//   RMSNormWithLearnableEpsilon, AdaptiveNormalization, BatchRenorm1d.

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
    // ---- RMSNormWithLearnableEpsilon ----
    if let Some(rms) = layer.as_rms_norm_learnable_eps() {
        let features = rms.features;
        let params_len = 2 * features;
        let params_view =
            MatrixBufferView::new(params_handle.clone(), slice.start, params_len);
        let out_handle = gpu.allocate_gpu_matrix_handle(input.rows(), features);
        gpu.run_rms_norm_learnable_eps_forward_buffered_handle(
            input,
            &params_view,
            &out_handle,
        );
        let ctx = DynamicContext::Buffered(
            BufferedContext::RMSNormWithLearnableEpsilon {
                input: input.clone(),
            },
        );
        return Some((out_handle, ctx));
    }

    // ---- AdaptiveNormalization ----
    if let Some(adnorm) = layer.as_adaptive_normalization() {
        let features = adnorm.features;
        let params_len = 7 * features;
        let params_view =
            MatrixBufferView::new(params_handle.clone(), slice.start, params_len);
        let out_handle = gpu.allocate_gpu_matrix_handle(input.rows(), features);
        gpu.run_adaptive_norm_forward_buffered_handle(
            input,
            &params_view,
            &out_handle,
        );
        let ctx = DynamicContext::Buffered(BufferedContext::AdaptiveNormalization {
            input: input.clone(),
        });
        return Some((out_handle, ctx));
    }

    // ---- BatchRenorm1d ----
    if let Some(bn) = layer.as_batch_renorm() {
        let features = bn.features;
        let params_len = 4 * features;
        let params_view =
            MatrixBufferView::new(params_handle.clone(), slice.start, params_len);
        let out_handle = gpu.allocate_gpu_matrix_handle(input.rows(), features);
        gpu.run_batch_renorm_forward_buffered_handle(
            input,
            &params_view,
            &out_handle,
        );
        let ctx = DynamicContext::Buffered(BufferedContext::BatchRenorm {
            input: input.clone(),
            mean: Vec::new(),
            var: Vec::new(),
            use_batch_stats: true,
        });
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
    // ---- RMSNormWithLearnableEpsilon ----
    if let Some(rms) = layer.as_rms_norm_learnable_eps() {
        let features = rms.features;
        let params_len = 2 * features;
        let DynamicContext::Buffered(bc) = ctx;
        let input_handle = match bc {
            BufferedContext::RMSNormWithLearnableEpsilon { input } => input.clone(),
            _ => panic!("Expected RMSNormWithLearnableEpsilon Buffered context"),
        };
        let params_view =
            MatrixBufferView::new(params_handle.clone(), slice.start, params_len);
        let grad_params_view =
            MatrixBufferView::new(grad_params_handle.clone(), slice.start, params_len);
        let gi = gpu.allocate_gpu_matrix_handle(grad_output.rows(), features);
        gpu.run_rms_norm_learnable_eps_backward_buffered_handle(
            &input_handle,
            grad_output,
            &params_view,
            &gi,
            &grad_params_view,
        );
        return Some(gi);
    }

    // ---- AdaptiveNormalization ----
    if let Some(adnorm) = layer.as_adaptive_normalization() {
        let features = adnorm.features;
        let params_len = 7 * features;
        let DynamicContext::Buffered(bc) = ctx;
        let input_handle = match bc {
            BufferedContext::AdaptiveNormalization { input } => input.clone(),
            _ => panic!("Expected AdaptiveNormalization Buffered context"),
        };
        let params_view =
            MatrixBufferView::new(params_handle.clone(), slice.start, params_len);
        let grad_params_view =
            MatrixBufferView::new(grad_params_handle.clone(), slice.start, params_len);
        let gi = gpu.allocate_gpu_matrix_handle(grad_output.rows(), features);
        gpu.run_adaptive_norm_backward_buffered_handle(
            &input_handle,
            grad_output,
            &params_view,
            &gi,
            &grad_params_view,
        );
        return Some(gi);
    }

    // ---- BatchRenorm1d ----
    if let Some(bn) = layer.as_batch_renorm() {
        let features = bn.features;
        let params_len = 4 * features;
        let DynamicContext::Buffered(bc) = ctx;
        let input_handle = match bc {
            BufferedContext::BatchRenorm { input, .. } => input.clone(),
            _ => panic!("Expected BatchRenorm Buffered context"),
        };
        let params_view =
            MatrixBufferView::new(params_handle.clone(), slice.start, params_len);
        let grad_params_view =
            MatrixBufferView::new(grad_params_handle.clone(), slice.start, params_len);
        let gi = gpu.allocate_gpu_matrix_handle(grad_output.rows(), features);
        gpu.run_batch_renorm_backward_buffered_handle(
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