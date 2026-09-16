// src/compute_manager/gpu/processor/layers/recurrent.rs
//
// Категория «рекуррентные слои»: IndRNN, Mamba.

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
    // ---- IndRNN ----
    if let Some(ind) = layer.as_ind_rnn() {
        let seq_len = ind.seq_len;
        let input_dim = ind.input_dim;
        let batch = input.rows();
        let h_all =
            gpu.allocate_gpu_matrix_handle(batch * seq_len * input_dim, 1);
        let params_len = input_dim * input_dim + 2 * input_dim;
        let params_view =
            MatrixBufferView::new(params_handle.clone(), slice.start, params_len);
        let out_handle =
            gpu.allocate_gpu_matrix_handle(batch, seq_len * input_dim);
        gpu.run_ind_rnn_forward_buffered_handle(
            input,
            &params_view,
            &out_handle,
            seq_len,
            input_dim,
            &h_all,
        );
        let ctx = DynamicContext::Buffered(BufferedContext::IndRNN {
            input: input.clone(),
            h_all,
        });
        return Some((out_handle, ctx));
    }

    // ---- Mamba ----
    if let Some(mamba) = layer.as_mamba() {
        let seq_len = mamba.seq_len;
        let input_dim = mamba.input_dim;
        let state_dim = mamba.state_dim;
        let batch = input.rows();
        let h_all =
            gpu.allocate_gpu_matrix_handle(batch * seq_len * state_dim, 1);
        let params_len =
            state_dim * state_dim + state_dim * input_dim + input_dim * state_dim + 2;
        let params_view =
            MatrixBufferView::new(params_handle.clone(), slice.start, params_len);
        let out_handle =
            gpu.allocate_gpu_matrix_handle(batch, seq_len * input_dim);
        gpu.run_mamba_forward_buffered_handle(
            input,
            &params_view,
            &out_handle,
            seq_len,
            input_dim,
            state_dim,
            &h_all,
        );

        // a_bar/b_bar на GPU-backward пересчитываются из параметров
        // слоя; в контекст кладём заглушки для согласованности с
        // BufferedContext::Mamba.
        let a_bar = gpu.allocate_gpu_matrix_handle(state_dim, state_dim);
        let b_bar = gpu.allocate_gpu_matrix_handle(state_dim, input_dim);

        let ctx = DynamicContext::Buffered(BufferedContext::Mamba {
            input: input.clone(),
            h_all,
            a_bar,
            b_bar,
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
    // ---- IndRNN ----
    if let Some(ind) = layer.as_ind_rnn() {
        let seq_len = ind.seq_len;
        let input_dim = ind.input_dim;
        let DynamicContext::Buffered(bc) = ctx;
        let (input_handle, h_all_handle) = match bc {
            BufferedContext::IndRNN { input, h_all } => {
                (input.clone(), h_all.clone())
            }
            _ => panic!("Expected IndRNN Buffered context"),
        };
        let params_len = input_dim * input_dim + 2 * input_dim;
        let params_view =
            MatrixBufferView::new(params_handle.clone(), slice.start, params_len);
        let grad_params_view =
            MatrixBufferView::new(grad_params_handle.clone(), slice.start, params_len);
        let gi = gpu
            .allocate_gpu_matrix_handle(grad_output.rows(), seq_len * input_dim);
        gpu.run_ind_rnn_backward_buffered_handle(
            &input_handle,
            grad_output,
            &params_view,
            &gi,
            &grad_params_view,
            seq_len,
            input_dim,
            &h_all_handle,
        );
        return Some(gi);
    }

    // ---- Mamba ----
    if let Some(mamba) = layer.as_mamba() {
        let seq_len = mamba.seq_len;
        let input_dim = mamba.input_dim;
        let state_dim = mamba.state_dim;
        let DynamicContext::Buffered(bc) = ctx;
        let (input_handle, h_all_handle) = match bc {
            BufferedContext::Mamba { input, h_all, .. } => {
                (input.clone(), h_all.clone())
            }
            _ => panic!("Expected Mamba Buffered context"),
        };
        let params_len =
            state_dim * state_dim + state_dim * input_dim + input_dim * state_dim + 2;
        let params_view =
            MatrixBufferView::new(params_handle.clone(), slice.start, params_len);
        let grad_params_view =
            MatrixBufferView::new(grad_params_handle.clone(), slice.start, params_len);
        let gi = gpu
            .allocate_gpu_matrix_handle(grad_output.rows(), seq_len * input_dim);
        gpu.run_mamba_backward_buffered_handle(
            &input_handle,
            grad_output,
            &params_view,
            &gi,
            &grad_params_view,
            seq_len,
            input_dim,
            state_dim,
            &h_all_handle,
        );
        return Some(gi);
    }

    None
}