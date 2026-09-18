// src/compute_manager/gpu/processor/layers/spectral_norm.rs
//
// Категория «спектральная нормализация»: SpectrallyNormalizedLinear.

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
    let sn = layer.as_spectral_norm_linear()?;
    let in_feat = sn.in_features;
    let out_feat = sn.out_features;
    let params_len = in_feat * out_feat + out_feat + 1;
    let params_view =
        MatrixBufferView::new(params_handle.clone(), slice.start, params_len);

    // Временные GPU буферы для состояния (u, v, sigma).
    let u_state = gpu.allocate_gpu_matrix_handle(in_feat, 1);
    let v_state = gpu.allocate_gpu_matrix_handle(out_feat, 1);
    let sigma_state = gpu.allocate_gpu_matrix_handle(1, 1);
    gpu.fill_gpu_handle(&u_state, 1.0);
    gpu.fill_gpu_handle(&v_state, 1.0);
    gpu.fill_gpu_handle(&sigma_state, 1.0);

    // Извлекаем scale из параметров (последний элемент).
    let scale_view = MatrixBufferView::new(
        params_handle.clone(),
        slice.start + in_feat * out_feat + out_feat,
        1,
    );
    let scale_cpu_handle =
        gpu.download_gpu_handle_to_cpu_handle(scale_view.parent_handle());
    let scale = {
        let guard = scale_cpu_handle.read();
        guard.as_slice().unwrap()[scale_view.offset_elements()]
    };

    let out_handle = gpu.allocate_gpu_matrix_handle(input.rows(), out_feat);
    gpu.run_spectral_norm_linear_forward_buffered_handle(
        input,
        &params_view,
        scale,
        &out_handle,
        &u_state,
        &v_state,
        &sigma_state,
    );

    // Состояние степенного метода теперь живёт в контексте —
    // слою больше не нужен set_last_sigma.
    let ctx = DynamicContext::Buffered(BufferedContext::SpectralNormLinear {
        input: input.clone(),
        u_state,
        v_state,
        sigma_state,
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
    let sn = layer.as_spectral_norm_linear()?;
    let in_feat = sn.in_features;
    let out_feat = sn.out_features;
    let params_len = in_feat * out_feat + out_feat + 1;

    let DynamicContext::Buffered(bc) = ctx;
    let (input_handle, sigma_state) = match bc {
        BufferedContext::SpectralNormLinear {
            input, sigma_state, ..
        } => (input.clone(), sigma_state.clone()),
        _ => panic!("Expected SpectralNormLinear Buffered context"),
    };

    let params_view =
        MatrixBufferView::new(params_handle.clone(), slice.start, params_len);
    let grad_params_view =
        MatrixBufferView::new(grad_params_handle.clone(), slice.start, params_len);
    let gi = gpu.allocate_gpu_matrix_handle(grad_output.rows(), in_feat);

    // Извлекаем scale из параметров.
    let scale_view = MatrixBufferView::new(
        params_handle.clone(),
        slice.start + in_feat * out_feat + out_feat,
        1,
    );
    let scale_cpu_handle =
        gpu.download_gpu_handle_to_cpu_handle(scale_view.parent_handle());
    let scale = {
        let guard = scale_cpu_handle.read();
        guard.as_slice().unwrap()[scale_view.offset_elements()]
    };

    // sigma читаем из per-chunk state-буфера (слой больше не хранит
    // last_sigma).
    let sigma = {
        let vec = gpu.download_gpu_handle_to_vec(&sigma_state);
        vec[0]
    };

    gpu.run_spectral_norm_linear_backward_buffered_handle(
        &input_handle,
        grad_output,
        &params_view,
        &gi,
        &grad_params_view,
        scale,
        sigma,
    );
    Some(gi)
}