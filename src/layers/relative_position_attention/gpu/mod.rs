// src/layers/relative_position_attention/gpu/mod.rs

pub mod pipeline;

use crate::compute_manager::gpu::compute::GpuCompute;
use crate::compute_manager::matrix_buffer::view::MatrixBufferView;
use crate::compute_manager::matrix_buffer::MatrixBufferHandle;
use vulkano::buffer::Subbuffer;

/// Вспомогательная функция: получает `Subbuffer<[f32]>` из `MatrixBufferView`.
fn subbuffer_from_view(gpu: &GpuCompute, view: &MatrixBufferView) -> Subbuffer<[f32]> {
    let parent_sub = gpu.get_gpu_subbuffer_from_handle(view.parent_handle());
    let start = view.offset_elements() as u64;
    let end = (view.offset_elements() + view.len()) as u64;
    parent_sub.slice(start..end)
}

impl GpuCompute {
    /// Прямой проход RelativePositionAttention на GPU (column-major).
    ///
    /// Все промежуточные тензоры хранятся в column-major:
    ///  - q, k, v:        (batch, seq_len * d_model)
    ///  - scores, weights: (batch, seq_len * seq_len)
    ///
    /// Параметры (Wq, bq, Wk, bk, Wv, bv, Wo, bo, rel_bias) передаются
    /// через плоский `params`. Веса — row-major, смещения и rel_bias — линейные.
    ///
    /// Промежуточные буферы (q_buf, k_buf, v_buf, scores_buf, weights_buf)
    /// передаются вызывающим кодом и сохраняются в
    /// `BufferedContext::RelativePositionAttention` для последующего
    /// обратного прохода — состояние слоя (RwLock<...State>) в GPU-пути
    /// не используется.
    pub fn run_relative_position_attention_forward_buffered_handle(
        &self,
        input: &MatrixBufferHandle,
        params: &MatrixBufferView,
        output: &MatrixBufferHandle,
        seq_len: usize,
        d_model: usize,
        q_buf: &MatrixBufferHandle,
        k_buf: &MatrixBufferHandle,
        v_buf: &MatrixBufferHandle,
        scores_buf: &MatrixBufferHandle,
        weights_buf: &MatrixBufferHandle,
    ) {
        assert!(input.is_gpu(), "Input handle must be GPU");
        assert!(output.is_gpu(), "Output handle must be GPU");
        assert!(params.is_gpu(), "Params view must point to GPU buffer");

        let batch = input.rows();
        let features = seq_len * d_model;
        assert_eq!(input.cols(), features, "Input cols mismatch");
        assert_eq!(output.rows(), batch);
        assert_eq!(output.cols(), features, "Output cols mismatch");

        let param_len = 4 * (d_model * d_model + d_model) + (2 * seq_len - 1);
        assert_eq!(params.len(), param_len, "Params length mismatch");

        let token_count = batch * seq_len;
        let token_total = token_count * d_model;
        let scores_total = token_count * seq_len;

        for buf in [q_buf, k_buf, v_buf] {
            assert_eq!(
                buf.rows() * buf.cols(),
                token_total,
                "QKV buffer size mismatch"
            );
        }
        for buf in [scores_buf, weights_buf] {
            assert_eq!(
                buf.rows() * buf.cols(),
                scores_total,
                "Scores/weights buffer size mismatch"
            );
        }

        let d = d_model;
        let wq_off = 0usize;
        let bq_off = wq_off + d * d;
        let wk_off = bq_off + d;
        let bk_off = wk_off + d * d;
        let wv_off = bk_off + d;
        let bv_off = wv_off + d * d;
        let wo_off = bv_off + d;
        let bo_off = wo_off + d * d;
        let rel_bias_off = bo_off + d;

        let parent = params.parent_handle().clone();
        let base = params.offset_elements();

        let rel_bias_view = MatrixBufferView::new(
            parent.clone(),
            base + rel_bias_off,
            2 * seq_len - 1,
        );
        let wo_view = MatrixBufferView::with_shape(
            parent.clone(),
            base + wo_off,
            d * d,
            d,
            d,
        );
        let bo_view = MatrixBufferView::new(parent.clone(), base + bo_off, d);

        let push = [batch as u32, seq_len as u32, d_model as u32];

        // 1. Подготовка Q, K, V: три линейных преобразования за один dispatch.
        let prepare_pipeline = &self.relative_position_attention_pipelines().prepare_qkv;
        self.run_compute_shader(
            prepare_pipeline,
            &[
                (0, self.get_gpu_subbuffer_from_handle(input)),
                (1, subbuffer_from_view(self, params)),
                (2, self.get_gpu_subbuffer_from_handle(q_buf)),
                (3, self.get_gpu_subbuffer_from_handle(k_buf)),
                (4, self.get_gpu_subbuffer_from_handle(v_buf)),
            ],
            &push,
            token_total * 3,
        );

        // 2. Scores и softmax.
        let scores_pipeline = &self.relative_position_attention_pipelines().scores_softmax;
        self.run_compute_shader(
            scores_pipeline,
            &[
                (0, self.get_gpu_subbuffer_from_handle(q_buf)),
                (1, self.get_gpu_subbuffer_from_handle(k_buf)),
                (2, subbuffer_from_view(self, &rel_bias_view)),
                (3, self.get_gpu_subbuffer_from_handle(scores_buf)),
                (4, self.get_gpu_subbuffer_from_handle(weights_buf)),
            ],
            &push,
            token_count,
        );

        // 3. Выход: y = b_o + weights · V · W_o^T.
        let output_pipeline = &self.relative_position_attention_pipelines().output;
        self.run_compute_shader(
            output_pipeline,
            &[
                (0, self.get_gpu_subbuffer_from_handle(weights_buf)),
                (1, self.get_gpu_subbuffer_from_handle(v_buf)),
                (2, subbuffer_from_view(self, &wo_view)),
                (3, subbuffer_from_view(self, &bo_view)),
                (4, self.get_gpu_subbuffer_from_handle(output)),
            ],
            &push,
            token_total,
        );
    }

    /// Обратный проход RelativePositionAttention на GPU (column-major).
    ///
    /// Принимает сохранённые промежуточные буферы с forward. Градиенты по
    /// параметрам записываются в `grad_params` через атомарное накопление.
    /// Состояние слоя (RwLock<RelativePositionAttentionState>) в GPU-пути
    /// не используется — все промежуточные тензоры приходят через аргументы.
    pub fn run_relative_position_attention_backward_buffered_handle(
        &self,
        input: &MatrixBufferHandle,
        grad_out: &MatrixBufferHandle,
        params: &MatrixBufferView,
        grad_input: &MatrixBufferHandle,
        grad_params: &MatrixBufferView,
        seq_len: usize,
        d_model: usize,
        q_buf: &MatrixBufferHandle,
        k_buf: &MatrixBufferHandle,
        v_buf: &MatrixBufferHandle,
        _scores_buf: &MatrixBufferHandle,
        weights_buf: &MatrixBufferHandle,
    ) {
        assert!(input.is_gpu(), "Input handle must be GPU");
        assert!(grad_out.is_gpu(), "grad_out handle must be GPU");
        assert!(grad_input.is_gpu(), "grad_input handle must be GPU");
        assert!(grad_params.is_gpu(), "grad_params view must point to GPU buffer");
        assert!(params.is_gpu(), "Params view must point to GPU buffer");

        let batch = input.rows();
        let features = seq_len * d_model;
        let d = d_model;
        let token_count = batch * seq_len;
        let token_total = token_count * d;
        let scores_total = token_count * seq_len;

        assert_eq!(input.cols(), features, "Input cols mismatch");
        assert_eq!(grad_out.rows(), batch);
        assert_eq!(grad_out.cols(), features, "grad_out cols mismatch");
        assert_eq!(grad_input.rows(), batch);
        assert_eq!(grad_input.cols(), features, "grad_input cols mismatch");

        let param_len = 4 * (d * d + d) + (2 * seq_len - 1);
        assert_eq!(params.len(), param_len, "Params length mismatch");
        assert_eq!(grad_params.len(), param_len, "grad_params length mismatch");

        // 1. Обнуление grad_params.
        let zero_handle = self.upload_vec_to_gpu_handle(
            &vec![0.0f32; grad_params.len()],
            grad_params.len(),
            1,
        );
        self.copy_gpu_handle_region(
            &zero_handle,
            grad_params.parent_handle(),
            0,
            grad_params.offset_elements(),
            grad_params.len(),
        );

        // 2. Смещения параметров в плоском буфере.
        let wq_off = 0usize;
        let bq_off = wq_off + d * d;
        let wk_off = bq_off + d;
        let bk_off = wk_off + d * d;
        let wv_off = bk_off + d;
        let bv_off = wv_off + d * d;
        let wo_off = bv_off + d;
        let bo_off = wo_off + d * d;
        let rel_bias_off = bo_off + d;

        let parent = params.parent_handle().clone();
        let base = params.offset_elements();

        let wo_view = MatrixBufferView::with_shape(
            parent.clone(),
            base + wo_off,
            d * d,
            d,
            d,
        );
        let bo_view = MatrixBufferView::new(parent.clone(), base + bo_off, d);

        let grad_parent = grad_params.parent_handle().clone();
        let grad_base = grad_params.offset_elements();

        let grad_rel_bias_view = MatrixBufferView::new(
            grad_parent.clone(),
            grad_base + rel_bias_off,
            2 * seq_len - 1,
        );

        // 3. Временные буферы.
        let (attn_out_buf, attn_out_raw) = self.acquire_temp_buffer(token_total);
        let (d_attn_out_buf, d_attn_out_raw) = self.acquire_temp_buffer(token_total);
        let (d_weights_buf, d_weights_raw) = self.acquire_temp_buffer(scores_total);
        let (d_scores_buf, d_scores_raw) = self.acquire_temp_buffer(scores_total);
        let (d_q_buf, d_q_raw) = self.acquire_temp_buffer(token_total);
        let (d_k_buf, d_k_raw) = self.acquire_temp_buffer(token_total);
        let (d_v_buf, d_v_raw) = self.acquire_temp_buffer(token_total);

        let push = [batch as u32, seq_len as u32, d_model as u32];

        // 4. Вычисление attn_out = weights · V через shader `output`
        //    с единичной матрицей W_o и нулевым b_o.
        //    Так как y = b_o + Σ_s weights[r,t,s] · (Σ_i v[r,s,i]·W_o[j,i]),
        //    при W_o = I, b_o = 0 получаем y = Σ_s weights[r,t,s] · v[r,s,j] = attn_out.
        let mut identity = vec![0.0f32; d * d];
        for i in 0..d {
            identity[i * d + i] = 1.0;
        }
        let identity_handle = self.upload_vec_to_gpu_handle(&identity, d, d);
        let zero_b_o_handle = self.upload_vec_to_gpu_handle(&vec![0.0f32; d], d, 1);

        let output_pipeline = &self.relative_position_attention_pipelines().output;
        self.run_compute_shader(
            output_pipeline,
            &[
                (0, self.get_gpu_subbuffer_from_handle(weights_buf)),
                (1, self.get_gpu_subbuffer_from_handle(v_buf)),
                (2, self.get_gpu_subbuffer_from_handle(&identity_handle)),
                (3, self.get_gpu_subbuffer_from_handle(&zero_b_o_handle)),
                (4, attn_out_buf.clone()),
            ],
            &push,
            token_total,
        );

        // 5. Обратный проход через выходной линейный слой:
        //    d_attn_out = go · W_o, grad_Wo, grad_bo.
        let bwd_out_pipeline =
            &self.relative_position_attention_pipelines().backward_output_params;
        self.run_compute_shader(
            bwd_out_pipeline,
            &[
                (0, self.get_gpu_subbuffer_from_handle(grad_out)),
                (1, attn_out_buf.clone()),
                (2, subbuffer_from_view(self, &wo_view)),
                (3, subbuffer_from_view(self, &bo_view)),
                (4, d_attn_out_buf.clone()),
                (5, subbuffer_from_view(self, grad_params)),
            ],
            &push,
            token_count,
        );

        // 6. Обратный проход через V и weights: d_v, d_weights.
        let bwd_vw_pipeline =
            &self.relative_position_attention_pipelines().backward_values_weights;
        self.run_compute_shader(
            bwd_vw_pipeline,
            &[
                (0, d_attn_out_buf.clone()),
                (1, self.get_gpu_subbuffer_from_handle(weights_buf)),
                (2, self.get_gpu_subbuffer_from_handle(v_buf)),
                (3, d_v_buf.clone()),
                (4, d_weights_buf.clone()),
            ],
            &push,
            token_total + scores_total,
        );

        // 7. Обратный проход через softmax: d_scores.
        let bwd_ss_pipeline =
            &self.relative_position_attention_pipelines().backward_scores_softmax;
        self.run_compute_shader(
            bwd_ss_pipeline,
            &[
                (0, self.get_gpu_subbuffer_from_handle(weights_buf)),
                (1, d_weights_buf.clone()),
                (2, d_scores_buf.clone()),
            ],
            &push,
            token_count,
        );

        // 8. Обратный проход через Q, K и rel_bias: d_q, d_k, grad_rel_bias.
        let bwd_qkv_pipeline =
            &self.relative_position_attention_pipelines().backward_qkv_params;
        self.run_compute_shader(
            bwd_qkv_pipeline,
            &[
                (0, self.get_gpu_subbuffer_from_handle(q_buf)),
                (1, self.get_gpu_subbuffer_from_handle(k_buf)),
                (2, d_scores_buf.clone()),
                (3, d_q_buf.clone()),
                (4, d_k_buf.clone()),
                (5, subbuffer_from_view(self, &grad_rel_bias_view)),
            ],
            &push,
            token_total * 2 + scores_total,
        );

        // 9. Обратный проход через QKV-проекции к входу и к их параметрам.
        let bwd_in_pipeline =
            &self.relative_position_attention_pipelines().backward_input_params;
        self.run_compute_shader(
            bwd_in_pipeline,
            &[
                (0, self.get_gpu_subbuffer_from_handle(input)),
                (1, d_q_buf.clone()),
                (2, d_k_buf.clone()),
                (3, d_v_buf.clone()),
                (4, subbuffer_from_view(self, params)),
                (5, self.get_gpu_subbuffer_from_handle(grad_input)),
                (6, subbuffer_from_view(self, grad_params)),
            ],
            &push,
            token_total,
        );

        // 10. Освобождение временных буферов.
        self.release_temp_buffer(attn_out_buf, attn_out_raw);
        self.release_temp_buffer(d_attn_out_buf, d_attn_out_raw);
        self.release_temp_buffer(d_weights_buf, d_weights_raw);
        self.release_temp_buffer(d_scores_buf, d_scores_raw);
        self.release_temp_buffer(d_q_buf, d_q_raw);
        self.release_temp_buffer(d_k_buf, d_k_raw);
        self.release_temp_buffer(d_v_buf, d_v_raw);
    }
}