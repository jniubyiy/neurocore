// src/layers/per_feature_attention/gpu/mod.rs

pub mod pipeline;

use crate::compute_manager::gpu::compute::GpuCompute;
use crate::compute_manager::matrix_buffer::view::MatrixBufferView;
use crate::compute_manager::matrix_buffer::MatrixBufferHandle;
use vulkano::buffer::Subbuffer;

fn subbuffer_from_view(gpu: &GpuCompute, view: &MatrixBufferView) -> Subbuffer<[f32]> {
    let parent_sub = gpu.get_gpu_subbuffer_from_handle(view.parent_handle());
    let start = view.offset_elements() as u64;
    let end = (view.offset_elements() + view.len()) as u64;
    parent_sub.slice(start..end)
}

impl GpuCompute {
    /// Прямой проход PerFeatureAttention на GPU.
    ///
    /// Все промежуточные буферы (q_raw, k_raw, v_raw, q_phi, k_phi, kv, z,
    /// attn, denom) аллоцируются в `attention.rs::forward` и передаются сюда
    /// для заполнения. Они сохраняются в контексте слоя.
    pub fn run_per_feature_attention_forward_buffered_handle(
        &self,
        input: &MatrixBufferHandle,
        params: &MatrixBufferView,
        output: &MatrixBufferHandle,
        seq_len: usize,
        d_model: usize,
        d_head: usize,
        q_raw: &MatrixBufferHandle,
        k_raw: &MatrixBufferHandle,
        v_raw: &MatrixBufferHandle,
        q_phi: &MatrixBufferHandle,
        k_phi: &MatrixBufferHandle,
        kv: &MatrixBufferHandle,
        z: &MatrixBufferHandle,
        attn: &MatrixBufferHandle,
        denom: &MatrixBufferHandle,
    ) {
        assert!(input.is_gpu(), "Input handle must be GPU");
        assert!(output.is_gpu(), "Output handle must be GPU");
        assert!(params.is_gpu(), "Params view must point to GPU buffer");

        let batch = input.rows();
        let features = seq_len * d_model;
        assert_eq!(input.cols(), features, "Input cols mismatch");
        assert_eq!(output.rows(), batch);
        assert_eq!(output.cols(), features, "Output cols mismatch");

        let param_len = d_model * (10 * d_head + 2);
        assert_eq!(params.len(), param_len, "Params length mismatch");

        let token_total = batch * seq_len * d_model;
        let dh_total = token_total * d_head;
        let kv_total = batch * d_model * d_head * d_head;
        let z_total = batch * d_model * d_head;

        for h in [q_raw, k_raw, v_raw, q_phi, k_phi, attn] {
            assert_eq!(h.rows() * h.cols(), dh_total, "dh buffer size mismatch");
        }
        assert_eq!(kv.rows() * kv.cols(), kv_total, "kv size mismatch");
        assert_eq!(z.rows() * z.cols(), z_total, "z size mismatch");
        assert_eq!(denom.rows() * denom.cols(), token_total, "denom size mismatch");

        let push = [
            batch as u32,
            seq_len as u32,
            d_model as u32,
            d_head as u32,
        ];

        let in_buf = self.get_gpu_subbuffer_from_handle(input);
        let params_buf = subbuffer_from_view(self, params);
        let out_buf = self.get_gpu_subbuffer_from_handle(output);

        let pipelines = self.per_feature_attention_pipelines();

        // F1: QKV + φ.
        self.run_compute_shader(
            &pipelines.fwd_qkv,
            &[
                (0, in_buf.clone()),
                (1, params_buf.clone()),
                (2, self.get_gpu_subbuffer_from_handle(q_raw)),
                (3, self.get_gpu_subbuffer_from_handle(k_raw)),
                (4, self.get_gpu_subbuffer_from_handle(v_raw)),
                (5, self.get_gpu_subbuffer_from_handle(q_phi)),
                (6, self.get_gpu_subbuffer_from_handle(k_phi)),
            ],
            &push,
            token_total,
        );

        // F2: kv и z (reduction по t).
        let kvz_total = batch * d_model * (d_head * d_head + d_head);
        self.run_compute_shader(
            &pipelines.fwd_kvz,
            &[
                (0, self.get_gpu_subbuffer_from_handle(k_phi)),
                (1, self.get_gpu_subbuffer_from_handle(v_raw)),
                (2, self.get_gpu_subbuffer_from_handle(kv)),
                (3, self.get_gpu_subbuffer_from_handle(z)),
            ],
            &push,
            kvz_total,
        );

        // F3: denom, attention, выход.
        self.run_compute_shader(
            &pipelines.fwd_out,
            &[
                (0, self.get_gpu_subbuffer_from_handle(q_phi)),
                (1, self.get_gpu_subbuffer_from_handle(v_raw)),
                (2, self.get_gpu_subbuffer_from_handle(kv)),
                (3, self.get_gpu_subbuffer_from_handle(z)),
                (4, params_buf),
                (5, out_buf),
                (6, self.get_gpu_subbuffer_from_handle(denom)),
                (7, self.get_gpu_subbuffer_from_handle(attn)),
            ],
            &push,
            token_total,
        );
    }

    /// Обратный проход PerFeatureAttention на GPU.
    ///
    /// `grad_params` должен быть предварительно обнулён вызывающим кодом
    /// (в `process_backward_gpu_buffered` это делается через `fill_gpu_handle`).
    #[allow(clippy::too_many_arguments)]
    pub fn run_per_feature_attention_backward_buffered_handle(
        &self,
        input: &MatrixBufferHandle,
        grad_out: &MatrixBufferHandle,
        params: &MatrixBufferView,
        grad_input: &MatrixBufferHandle,
        grad_params: &MatrixBufferView,
        seq_len: usize,
        d_model: usize,
        d_head: usize,
        q_raw: &MatrixBufferHandle,
        k_raw: &MatrixBufferHandle,
        v_raw: &MatrixBufferHandle,
        q_phi: &MatrixBufferHandle,
        k_phi: &MatrixBufferHandle,
        kv: &MatrixBufferHandle,
        z: &MatrixBufferHandle,
        attn: &MatrixBufferHandle,
        denom: &MatrixBufferHandle,
    ) {
        assert!(input.is_gpu(), "Input handle must be GPU");
        assert!(grad_out.is_gpu(), "grad_out handle must be GPU");
        assert!(grad_input.is_gpu(), "grad_input handle must be GPU");
        assert!(grad_params.is_gpu(), "grad_params view must point to GPU buffer");
        assert!(params.is_gpu(), "Params view must point to GPU buffer");

        let batch = input.rows();
        let features = seq_len * d_model;
        assert_eq!(input.cols(), features, "Input cols mismatch");
        assert_eq!(grad_out.rows(), batch);
        assert_eq!(grad_out.cols(), features, "grad_out cols mismatch");
        assert_eq!(grad_input.rows(), batch);
        assert_eq!(grad_input.cols(), features, "grad_input cols mismatch");

        let param_len = d_model * (10 * d_head + 2);
        assert_eq!(params.len(), param_len, "Params length mismatch");
        assert_eq!(grad_params.len(), param_len, "grad_params length mismatch");

        let token_total = batch * seq_len * d_model;
        let dh_total = token_total * d_head;
        let kv_total = batch * d_model * d_head * d_head;
        let z_total = batch * d_model * d_head;

        let push = [
            batch as u32,
            seq_len as u32,
            d_model as u32,
            d_head as u32,
        ];

        let in_buf = self.get_gpu_subbuffer_from_handle(input);
        let go_buf = self.get_gpu_subbuffer_from_handle(grad_out);
        let gi_buf = self.get_gpu_subbuffer_from_handle(grad_input);
        let params_buf = subbuffer_from_view(self, params);
        let grad_params_buf = subbuffer_from_view(self, grad_params);

        // Временные буферы для backward.
        let (grad_attn_buf, grad_attn_raw) = self.acquire_temp_buffer(dh_total);
        let (grad_num_buf, grad_num_raw) = self.acquire_temp_buffer(dh_total);
        let (grad_denom_buf, grad_denom_raw) = self.acquire_temp_buffer(token_total);
        let (grad_kv_buf, grad_kv_raw) = self.acquire_temp_buffer(kv_total);
        let (grad_z_buf, grad_z_raw) = self.acquire_temp_buffer(z_total);
        let (grad_q_phi_buf, grad_q_phi_raw) = self.acquire_temp_buffer(dh_total);
        let (grad_v_raw_buf, grad_v_raw_raw) = self.acquire_temp_buffer(dh_total);
        let (grad_q_raw_buf, grad_q_raw_raw) = self.acquire_temp_buffer(dh_total);
        let (grad_k_raw_buf, grad_k_raw_raw) = self.acquire_temp_buffer(dh_total);

        let pipelines = self.per_feature_attention_pipelines();

        // B1: grad_bo, grad_wo, grad_sb (atomic); grad_attn, grad_num, grad_denom (write).
        self.run_compute_shader(
            &pipelines.bwd_out,
            &[
                (0, go_buf),
                (1, self.get_gpu_subbuffer_from_handle(attn)),
                (2, self.get_gpu_subbuffer_from_handle(v_raw)),
                (3, self.get_gpu_subbuffer_from_handle(denom)),
                (4, params_buf.clone()),
                (5, grad_params_buf.clone()),
                (6, grad_attn_buf.clone()),
                (7, grad_num_buf.clone()),
                (8, grad_denom_buf.clone()),
            ],
            &push,
            token_total,
        );

        // B2: grad_kv, grad_z (reduction по t).
        self.run_compute_shader(
            &pipelines.bwd_grad_kvz,
            &[
                (0, grad_num_buf.clone()),
                (1, grad_denom_buf.clone()),
                (2, self.get_gpu_subbuffer_from_handle(q_phi)),
                (3, grad_kv_buf.clone()),
                (4, grad_z_buf.clone()),
            ],
            &push,
            z_total,
        );

        // B3: grad_q_phi.
        self.run_compute_shader(
            &pipelines.bwd_grad_q_phi,
            &[
                (0, grad_num_buf.clone()),
                (1, grad_denom_buf.clone()),
                (2, self.get_gpu_subbuffer_from_handle(kv)),
                (3, self.get_gpu_subbuffer_from_handle(z)),
                (4, grad_q_phi_buf.clone()),
            ],
            &push,
            dh_total,
        );

        // B4: grad_v_raw.
        self.run_compute_shader(
            &pipelines.bwd_grad_v,
            &[
                (0, grad_num_buf.clone()),
                (1, grad_kv_buf.clone()),
                (2, self.get_gpu_subbuffer_from_handle(k_phi)),
                (3, params_buf.clone()),
                (4, grad_v_raw_buf.clone()),
            ],
            &push,
            dh_total,
        );

        // B5: grad_q_raw, grad_k_raw (через φ').
        self.run_compute_shader(
            &pipelines.bwd_qk_raw,
            &[
                (0, grad_q_phi_buf.clone()),
                (1, grad_kv_buf.clone()),
                (2, grad_z_buf.clone()),
                (3, self.get_gpu_subbuffer_from_handle(v_raw)),
                (4, self.get_gpu_subbuffer_from_handle(q_raw)),
                (5, self.get_gpu_subbuffer_from_handle(k_raw)),
                (6, grad_q_raw_buf.clone()),
                (7, grad_k_raw_buf.clone()),
            ],
            &push,
            dh_total,
        );

        // B6: grad_Wq/bq/Wk/bk/Wv/bv (atomic).
        self.run_compute_shader(
            &pipelines.bwd_qkv_params,
            &[
                (0, grad_q_raw_buf.clone()),
                (1, grad_k_raw_buf.clone()),
                (2, grad_v_raw_buf.clone()),
                (3, in_buf),
                (4, grad_params_buf),
            ],
            &push,
            z_total,
        );

        // B7: grad_x (прямая запись).
        self.run_compute_shader(
            &pipelines.bwd_grad_x,
            &[
                (0, grad_q_raw_buf.clone()),
                (1, grad_k_raw_buf.clone()),
                (2, grad_v_raw_buf.clone()),
                (3, params_buf),
                (4, gi_buf),
            ],
            &push,
            token_total,
        );

        // Освобождение временных буферов.
        self.release_temp_buffer(grad_attn_buf, grad_attn_raw);
        self.release_temp_buffer(grad_num_buf, grad_num_raw);
        self.release_temp_buffer(grad_denom_buf, grad_denom_raw);
        self.release_temp_buffer(grad_kv_buf, grad_kv_raw);
        self.release_temp_buffer(grad_z_buf, grad_z_raw);
        self.release_temp_buffer(grad_q_phi_buf, grad_q_phi_raw);
        self.release_temp_buffer(grad_v_raw_buf, grad_v_raw_raw);
        self.release_temp_buffer(grad_q_raw_buf, grad_q_raw_raw);
        self.release_temp_buffer(grad_k_raw_buf, grad_k_raw_raw);
    }
}