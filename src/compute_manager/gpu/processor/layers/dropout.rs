// src/compute_manager/gpu/processor/layers/dropout.rs
//
// Категория «dropout»: ConcreteDropout, AdaptiveDropout.

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
    // ---- ConcreteDropout ----
    if let Some(cdrop) = layer.as_concrete_dropout() {
        let temperature = cdrop.temperature;
        let logit_view = MatrixBufferView::new(params_handle.clone(), slice.start, 1);
        let batch = input.rows();
        let arg_out = gpu.allocate_gpu_matrix_handle(batch * input.cols(), 1);
        let out_handle = gpu.allocate_gpu_matrix_handle(batch, input.cols());

        // FIX (стохастичность dropout на GPU):
        //
        // Раньше здесь использовалось `cdrop.seed as u32` — константа,
        // передаваемая в шейдер. Это давало одну и ту же маску на всех
        // forward-проходах, точно так же, как было на CPU до фикса.
        //
        // Теперь используется `cdrop.next_seed()` — атомарный счётчик
        // вызовов forward в структуре слоя. Счётчик общий с CPU-путём,
        // поэтому поведение согласовано: и на CPU, и на GPU каждый
        // forward получает свежий seed.
        //
        // Приведение u64 → u32 сохраняет нижние 32 бита — этого
        // достаточно для RNG шейдера (xorshift32). Коллизии возможны
        // раз в 2^32 вызовов, что при обучении недостижимо.
        let seed = cdrop.next_seed() as u32;
        gpu.run_concrete_dropout_forward_buffered_handle(
            input,
            &logit_view,
            temperature,
            &out_handle,
            &arg_out,
            seed,
        );
        let ctx = DynamicContext::Buffered(BufferedContext::ConcreteDropout {
            input: input.clone(),
            arg: arg_out,
        });
        return Some((out_handle, ctx));
    }

    // ---- AdaptiveDropout ----
    if let Some(adrop) = layer.as_adaptive_dropout() {
        let features = adrop.features;
        let params_len = 2 * features;
        let params_view =
            MatrixBufferView::new(params_handle.clone(), slice.start, params_len);
        let batch = input.rows();
        let mask_out = gpu.allocate_gpu_matrix_handle(batch, features);
        let arg_out = gpu.allocate_gpu_matrix_handle(batch, features);
        let out_handle = gpu.allocate_gpu_matrix_handle(batch, features);
        let seed = adrop.seed as u32;
        gpu.run_adaptive_dropout_forward_buffered_handle(
            input,
            &params_view,
            &out_handle,
            &mask_out,
            &arg_out,
            seed,
        );
        let ctx = DynamicContext::Buffered(BufferedContext::AdaptiveDropout {
            input: input.clone(),
            mask: mask_out,
            arg: arg_out,
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
    // ---- ConcreteDropout ----
    if let Some(cdrop) = layer.as_concrete_dropout() {
        let temperature = cdrop.temperature;
        let DynamicContext::Buffered(bc) = ctx;
        let (input_handle, arg_handle) = match bc {
            BufferedContext::ConcreteDropout { input, arg } => {
                (input.clone(), arg.clone())
            }
            _ => panic!("Expected ConcreteDropout Buffered context"),
        };
        let logit_view = MatrixBufferView::new(params_handle.clone(), slice.start, 1);
        let grad_logit_view =
            MatrixBufferView::new(grad_params_handle.clone(), slice.start, 1);
        let gi =
            gpu.allocate_gpu_matrix_handle(grad_output.rows(), grad_output.cols());
        gpu.run_concrete_dropout_backward_buffered_handle(
            &input_handle,
            grad_output,
            &logit_view,
            temperature,
            &arg_handle,
            &gi,
            &grad_logit_view,
        );
        return Some(gi);
    }

    // ---- AdaptiveDropout ----
    if let Some(adrop) = layer.as_adaptive_dropout() {
        let features = adrop.features;
        let params_len = 2 * features;
        let DynamicContext::Buffered(bc) = ctx;
        let (input_handle, mask_handle, arg_handle) = match bc {
            BufferedContext::AdaptiveDropout { input, mask, arg } => {
                (input.clone(), mask.clone(), arg.clone())
            }
            _ => panic!("Expected AdaptiveDropout Buffered context"),
        };
        let params_view =
            MatrixBufferView::new(params_handle.clone(), slice.start, params_len);
        let grad_params_view =
            MatrixBufferView::new(grad_params_handle.clone(), slice.start, params_len);
        let gi = gpu.allocate_gpu_matrix_handle(grad_output.rows(), features);
        gpu.run_adaptive_dropout_backward_buffered_handle(
            &input_handle,
            grad_output,
            &params_view,
            &mask_handle,
            &arg_handle,
            &gi,
            &grad_params_view,
        );
        return Some(gi);
    }

    None
}