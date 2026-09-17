// src/compute_manager/gpu/processor/layers/attention.rs
//
// Категория «внимание»: LinearAttention, RelativePositionAttention,
// PerFeatureAttention.

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
    // ---- LinearAttention ----
    if let Some(lin_att) = layer.as_linear_attention() {
        let seq_len = lin_att.seq_len;
        let d_model = lin_att.d_model;
        let d_head = lin_att.d_head();
        let max_heads = lin_att.max_heads;
        let batch = input.rows();
        let total_tokens = batch * seq_len;

        let q_raw = gpu.allocate_gpu_matrix_handle(total_tokens, d_model);
        let k_raw = gpu.allocate_gpu_matrix_handle(total_tokens, d_model);
        let v_raw = gpu.allocate_gpu_matrix_handle(total_tokens, d_model);
        let q_phi = gpu.allocate_gpu_matrix_handle(total_tokens, d_model);
        let k_phi = gpu.allocate_gpu_matrix_handle(total_tokens, d_model);

        let kv_size = max_heads * batch * d_head * d_head;
        let z_size = max_heads * batch * d_head;
        let kv = gpu.allocate_gpu_matrix_handle(kv_size, 1);
        let z = gpu.allocate_gpu_matrix_handle(z_size, 1);

        let param_len = lin_att.param_len();
        let params_view =
            MatrixBufferView::new(params_handle.clone(), slice.start, param_len);

        let out_handle =
            gpu.allocate_gpu_matrix_handle(input.rows(), lin_att.output_features());

        gpu.run_linear_attention_forward_buffered_handle_with_dims(
            input, &params_view, &out_handle,
            seq_len, d_model,
            &q_raw, &k_raw, &v_raw, &q_phi, &k_phi, &kv, &z,
        );

        let ctx = DynamicContext::Buffered(BufferedContext::LinearAttention {
            input: input.clone(),
            cpu_heads: Vec::new(),
            h_raw: 0.0,
            h_soft: 0.0,
            batch,
            seq: seq_len,
            d_model,
            d_head,
            q_raw: Some(q_raw),
            k_raw: Some(k_raw),
            v_raw: Some(v_raw),
            q_phi: Some(q_phi),
            k_phi: Some(k_phi),
            kv: Some(kv),
            z: Some(z),
        });
        return Some((out_handle, ctx));
    }

    // ---- RelativePositionAttention ----
    if let Some(rel_att) = layer.as_relative_position_attention() {
        let seq_len = rel_att.seq_len;
        let d_model = rel_att.d_model;
        let batch = input.rows();
        let total_tokens = batch * seq_len;

        let q = gpu.allocate_gpu_matrix_handle(total_tokens, d_model);
        let k = gpu.allocate_gpu_matrix_handle(total_tokens, d_model);
        let v = gpu.allocate_gpu_matrix_handle(total_tokens, d_model);
        let scores = gpu.allocate_gpu_matrix_handle(total_tokens, seq_len);
        let weights = gpu.allocate_gpu_matrix_handle(total_tokens, seq_len);

        let param_len = rel_att.param_len();
        let params_view =
            MatrixBufferView::new(params_handle.clone(), slice.start, param_len);

        let out_handle = gpu
            .allocate_gpu_matrix_handle(input.rows(), rel_att.output_features());

        gpu.run_relative_position_attention_forward_buffered_handle(
            input, &params_view, &out_handle,
            seq_len, d_model,
            &q, &k, &v, &scores, &weights,
        );

        let ctx = DynamicContext::Buffered(
            BufferedContext::RelativePositionAttention {
                input: input.clone(),
                q: Some(q),
                k: Some(k),
                v: Some(v),
                scores: Some(scores),
                weights: Some(weights),
                attn_out: None,
            },
        );
        return Some((out_handle, ctx));
    }

    // ---- PerFeatureAttention ----
    if let Some(pfa) = layer.as_per_feature_attention() {
        let seq_len = pfa.seq_len;
        let d_model = pfa.d_model;
        let d_head = pfa.d_head;
        let batch = input.rows();

        let token_total = batch * seq_len * d_model;
        let dh_total = token_total * d_head;
        let kv_total = batch * d_model * d_head * d_head;
        let z_total = batch * d_model * d_head;

        let q_raw = gpu.allocate_gpu_matrix_handle(dh_total, 1);
        let k_raw = gpu.allocate_gpu_matrix_handle(dh_total, 1);
        let v_raw = gpu.allocate_gpu_matrix_handle(dh_total, 1);
        let q_phi = gpu.allocate_gpu_matrix_handle(dh_total, 1);
        let k_phi = gpu.allocate_gpu_matrix_handle(dh_total, 1);
        let kv = gpu.allocate_gpu_matrix_handle(kv_total, 1);
        let z = gpu.allocate_gpu_matrix_handle(z_total, 1);
        let attn = gpu.allocate_gpu_matrix_handle(dh_total, 1);
        let denom = gpu.allocate_gpu_matrix_handle(token_total, 1);

        let param_len = pfa.param_len();
        let params_view =
            MatrixBufferView::new(params_handle.clone(), slice.start, param_len);

        let out_handle = gpu.allocate_gpu_matrix_handle(input.rows(), input.cols());

        gpu.run_per_feature_attention_forward_buffered_handle(
            input, &params_view, &out_handle,
            seq_len, d_model, d_head,
            &q_raw, &k_raw, &v_raw, &q_phi, &k_phi,
            &kv, &z, &attn, &denom,
        );

        let ctx = DynamicContext::Buffered(BufferedContext::PerFeatureAttentionGpu {
            input: input.clone(),
            q_raw, k_raw, v_raw, q_phi, k_phi,
            kv, z, attn, denom,
            batch, seq_len, d_model, d_head,
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
    // ---- LinearAttention ----
    if let Some(lin_att) = layer.as_linear_attention() {
        let seq_len = lin_att.seq_len;
        let d_model = lin_att.d_model;

        let DynamicContext::Buffered(bc) = ctx;
        let (input_handle, q_raw, k_raw, v_raw, q_phi, k_phi, kv, z) = match bc {
            BufferedContext::LinearAttention {
                input, q_raw, k_raw, v_raw, q_phi, k_phi, kv, z, ..
            } => (
                input.clone(),
                q_raw.clone().unwrap(),
                k_raw.clone().unwrap(),
                v_raw.clone().unwrap(),
                q_phi.clone().unwrap(),
                k_phi.clone().unwrap(),
                kv.clone().unwrap(),
                z.clone().unwrap(),
            ),
            _ => panic!("Expected LinearAttention Buffered context"),
        };

        let param_len = lin_att.param_len();
        let params_view =
            MatrixBufferView::new(params_handle.clone(), slice.start, param_len);
        let grad_params_view =
            MatrixBufferView::new(grad_params_handle.clone(), slice.start, param_len);

        let gi = gpu
            .allocate_gpu_matrix_handle(grad_output.rows(), lin_att.input_features());

        gpu.run_linear_attention_backward_buffered_handle_with_dims(
            &input_handle, grad_output, &params_view, &gi, &grad_params_view,
            seq_len, d_model,
            &q_raw, &k_raw, &v_raw, &q_phi, &k_phi, &kv, &z,
        );
        return Some(gi);
    }

    // ---- RelativePositionAttention ----
    if let Some(rel_att) = layer.as_relative_position_attention() {
        let seq_len = rel_att.seq_len;
        let d_model = rel_att.d_model;

        let DynamicContext::Buffered(bc) = ctx;
        let (input_handle, q, k, v, scores, weights) = match bc {
            BufferedContext::RelativePositionAttention {
                input, q, k, v, scores, weights, ..
            } => (
                input.clone(),
                q.clone().unwrap(),
                k.clone().unwrap(),
                v.clone().unwrap(),
                scores.clone().unwrap(),
                weights.clone().unwrap(),
            ),
            _ => panic!("Expected RelativePositionAttention Buffered context"),
        };

        let param_len = rel_att.param_len();
        let params_view =
            MatrixBufferView::new(params_handle.clone(), slice.start, param_len);
        let grad_params_view =
            MatrixBufferView::new(grad_params_handle.clone(), slice.start, param_len);

        let gi = gpu.allocate_gpu_matrix_handle(
            grad_output.rows(), rel_att.input_features(),
        );

        gpu.run_relative_position_attention_backward_buffered_handle(
            &input_handle, grad_output, &params_view, &gi, &grad_params_view,
            seq_len, d_model,
            &q, &k, &v, &scores, &weights,
        );
        return Some(gi);
    }

    // ---- PerFeatureAttention ----
    if let Some(pfa) = layer.as_per_feature_attention() {
        let seq_len = pfa.seq_len;
        let d_model = pfa.d_model;
        let d_head = pfa.d_head;

        let DynamicContext::Buffered(bc) = ctx;
        let (input_handle, q_raw, k_raw, v_raw, q_phi, k_phi, kv, z, attn, denom) =
            match bc {
                BufferedContext::PerFeatureAttentionGpu {
                    input,
                    q_raw, k_raw, v_raw, q_phi, k_phi,
                    kv, z, attn, denom, ..
                } => (
                    input.clone(),
                    q_raw.clone(),
                    k_raw.clone(),
                    v_raw.clone(),
                    q_phi.clone(),
                    k_phi.clone(),
                    kv.clone(),
                    z.clone(),
                    attn.clone(),
                    denom.clone(),
                ),
                _ => panic!("Expected PerFeatureAttentionGpu Buffered context"),
            };

        let param_len = pfa.param_len();
        let params_view =
            MatrixBufferView::new(params_handle.clone(), slice.start, param_len);
        let grad_params_view =
            MatrixBufferView::new(grad_params_handle.clone(), slice.start, param_len);

        let gi = gpu.allocate_gpu_matrix_handle(grad_output.rows(), pfa.input_features());

        gpu.run_per_feature_attention_backward_buffered_handle(
            &input_handle, grad_output, &params_view, &gi, &grad_params_view,
            seq_len, d_model, d_head,
            &q_raw, &k_raw, &v_raw, &q_phi, &k_phi,
            &kv, &z, &attn, &denom,
        );
        return Some(gi);
    }

    None
}