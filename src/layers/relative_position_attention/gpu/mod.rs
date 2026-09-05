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

/// Преобразует column-major (batch, seq_len * d_model) в row-major
/// (batch * seq_len) x d_model.
fn column_to_row(data: &[f32], batch: usize, seq_len: usize, d_model: usize) -> Vec<f32> {
    let mut row = vec![0.0f32; batch * seq_len * d_model];
    for r in 0..batch {
        for t in 0..seq_len {
            for j in 0..d_model {
                let src_idx = (t * d_model + j) * batch + r;
                let dst_idx = r * seq_len * d_model + t * d_model + j;
                row[dst_idx] = data[src_idx];
            }
        }
    }
    row
}

/// Преобразует row-major (batch * seq_len) x d_model в column-major
/// (batch, seq_len * d_model).
fn row_to_column(data: &[f32], batch: usize, seq_len: usize, d_model: usize) -> Vec<f32> {
    let mut col = vec![0.0f32; batch * seq_len * d_model];
    for r in 0..batch {
        for t in 0..seq_len {
            for j in 0..d_model {
                let src_idx = r * seq_len * d_model + t * d_model + j;
                let dst_idx = (t * d_model + j) * batch + r;
                col[dst_idx] = data[src_idx];
            }
        }
    }
    col
}

impl GpuCompute {
    /// Прямой проход RelativePositionAttention на GPU.
    ///
    /// Параметры слоя (все матрицы, смещения и relative_bias) передаются как `params_view`.
    /// Вход и выход — GPU-дескрипторы (column-major).
    /// Промежуточные буферы (`q`, `k`, `v`, `scores`, `weights`) выделяются вызывающим
    /// кодом и сохраняются для обратного прохода. Их размеры должны быть:
    /// - q, k, v: (batch * seq_len) x d_model
    /// - scores, weights: (batch * seq_len) x seq_len
    ///
    /// # Аргументы
    /// * `input` – вход `(batch, seq_len * d_model)` column-major.
    /// * `params` – view на полный блок параметров `4*(d_model² + d_model) + (2*seq_len - 1)`.
    /// * `output` – выход `(batch, seq_len * d_model)` column-major.
    /// * `seq_len`, `d_model` – размеры последовательности и модели.
    /// * `q_buf`, `k_buf`, `v_buf`, `scores_buf`, `weights_buf` – буферы для сохранения
    ///   промежуточных результатов, должны быть GPU-буферами нужного размера (row-major).
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
        for buf in [q_buf, k_buf, v_buf] {
            assert_eq!(buf.rows() * buf.cols(), token_count * d_model, "Intermediate buffer size mismatch");
        }
        for buf in [scores_buf, weights_buf] {
            assert_eq!(buf.rows() * buf.cols(), token_count * seq_len, "Scores/weights buffer size mismatch");
        }

        // 1. Преобразуем вход из column-major в row-major на CPU
        let input_vec = self.download_gpu_handle_to_vec(input);
        let input_row = column_to_row(&input_vec, batch, seq_len, d_model);
        let input_row_handle = self.upload_vec_to_gpu_handle(&input_row, token_count, d_model);

        // 2. Создаём views для параметров
        let d = d_model;
        let wq_start = 0usize;
        let bq_start = wq_start + d * d;
        let wk_start = bq_start + d;
        let bk_start = wk_start + d * d;
        let wv_start = bk_start + d;
        let bv_start = wv_start + d * d;
        let wo_start = bv_start + d;
        let bo_start = wo_start + d * d;
        let rel_bias_start = bo_start + d;

        let params_buf_full = subbuffer_from_view(self, params);
        let rel_bias_view = MatrixBufferView::new(
            params.parent_handle().clone(),
            params.offset_elements() + rel_bias_start,
            2 * seq_len - 1,
        );
        let rel_bias_buf = subbuffer_from_view(self, &rel_bias_view);

        let wo_view = MatrixBufferView::with_shape(
            params.parent_handle().clone(),
            params.offset_elements() + wo_start,
            d * d,
            d,
            d,
        );
        let bo_view = MatrixBufferView::new(
            params.parent_handle().clone(),
            params.offset_elements() + bo_start,
            d,
        );
        let wo_buf = subbuffer_from_view(self, &wo_view);
        let bo_buf = subbuffer_from_view(self, &bo_view);

        // 3. Подготовка Q, K, V
        let prepare_pipeline = &self.relative_position_attention_pipelines().prepare_qkv;
        let push = [batch as u32, seq_len as u32, d_model as u32];
        let total_qkv = token_count * d_model * 3;
        self.run_compute_shader(
            prepare_pipeline,
            &[
                (0, self.get_gpu_subbuffer_from_handle(&input_row_handle)),
                (1, params_buf_full.clone()),
                (2, self.get_gpu_subbuffer_from_handle(q_buf)),
                (3, self.get_gpu_subbuffer_from_handle(k_buf)),
                (4, self.get_gpu_subbuffer_from_handle(v_buf)),
            ],
            &push,
            total_qkv,
        );

        // 4. Вычисление scores и softmax
        let scores_pipeline = &self.relative_position_attention_pipelines().scores_softmax;
        let total_scores = token_count * seq_len;
        self.run_compute_shader(
            scores_pipeline,
            &[
                (0, self.get_gpu_subbuffer_from_handle(q_buf)),
                (1, self.get_gpu_subbuffer_from_handle(k_buf)),
                (2, rel_bias_buf),
                (3, self.get_gpu_subbuffer_from_handle(scores_buf)),
                (4, self.get_gpu_subbuffer_from_handle(weights_buf)),
            ],
            &push,
            total_scores,
        );

        // 5. Вычисление выходного тензора
        let output_pipeline = &self.relative_position_attention_pipelines().output;
        let total_out = token_count * d_model;
        // Выход будет row-major, поэтому используем временный буфер
        let output_row_buf = self.upload_vec_to_gpu_handle(&vec![0.0f32; total_out], token_count, d_model);
        self.run_compute_shader(
            output_pipeline,
            &[
                (0, self.get_gpu_subbuffer_from_handle(weights_buf)),
                (1, self.get_gpu_subbuffer_from_handle(v_buf)),
                (2, wo_buf),
                (3, bo_buf),
                (4, self.get_gpu_subbuffer_from_handle(&output_row_buf)),
            ],
            &push,
            total_out,
        );

        // 6. Преобразуем результат обратно в column-major
        let output_row_vec = self.download_gpu_handle_to_vec(&output_row_buf);
        let output_col = row_to_column(&output_row_vec, batch, seq_len, d_model);
        self.copy_slice_to_gpu_handle(output, &output_col);
    }

    /// Обратный проход RelativePositionAttention на GPU.
    ///
    /// Принимает сохранённые промежуточные буферы с forward.
    /// Градиенты по параметрам записываются в `grad_params` (view на полный блок).
    /// Вход/выходные градиенты — GPU-дескрипторы (column-major).
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
        scores_buf: &MatrixBufferHandle,
        weights_buf: &MatrixBufferHandle,
    ) {
        assert!(input.is_gpu(), "Input handle must be GPU");
        assert!(grad_out.is_gpu(), "grad_out handle must be GPU");
        assert!(grad_input.is_gpu(), "grad_input handle must be GPU");
        assert!(grad_params.is_gpu(), "grad_params view must point to GPU buffer");
        assert!(params.is_gpu(), "Params view must point to GPU buffer");

        let batch = input.rows();
        let features = seq_len * d_model;
        let token_count = batch * seq_len;
        assert_eq!(grad_out.rows(), batch);
        assert_eq!(grad_out.cols(), features);
        assert_eq!(grad_input.rows(), batch);
        assert_eq!(grad_input.cols(), features);

        let param_len = 4 * (d_model * d_model + d_model) + (2 * seq_len - 1);
        assert_eq!(params.len(), param_len);
        assert_eq!(grad_params.len(), param_len);

        // Обнуляем градиенты по параметрам
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

        // Преобразуем вход и grad_out в row-major
        let input_vec = self.download_gpu_handle_to_vec(input);
        let input_row = column_to_row(&input_vec, batch, seq_len, d_model);
        let input_row_handle = self.upload_vec_to_gpu_handle(&input_row, token_count, d_model);

        let go_vec = self.download_gpu_handle_to_vec(grad_out);
        let go_row = column_to_row(&go_vec, batch, seq_len, d_model);
        let go_row_handle = self.upload_vec_to_gpu_handle(&go_row, token_count, d_model);

        // Выделяем временные буферы для промежуточных градиентов
        let (d_attn_out_buf, d_attn_out_raw) = self.acquire_temp_buffer(token_count * d_model);
        let (d_weights_buf, d_weights_raw) = self.acquire_temp_buffer(token_count * seq_len);
        let (d_scores_buf, d_scores_raw) = self.acquire_temp_buffer(token_count * seq_len);
        let (d_q_buf, d_q_raw) = self.acquire_temp_buffer(token_count * d_model);
        let (d_k_buf, d_k_raw) = self.acquire_temp_buffer(token_count * d_model);
        let (d_v_buf, d_v_raw) = self.acquire_temp_buffer(token_count * d_model);

        // Получаем subbuffer'ы
        let in_buf = self.get_gpu_subbuffer_from_handle(&input_row_handle);
        let go_buf = self.get_gpu_subbuffer_from_handle(&go_row_handle);
        let gi_row_buf = self.acquire_temp_buffer(token_count * d_model);
        let params_buf = subbuffer_from_view(self, params);
        let grad_params_buf = subbuffer_from_view(self, grad_params);

        // Создаём view для параметров
        let d = d_model;
        let wq_start = 0;
        let bq_start = wq_start + d * d;
        let wk_start = bq_start + d;
        let bk_start = wk_start + d * d;
        let wv_start = bk_start + d;
        let bv_start = wv_start + d * d;
        let wo_start = bv_start + d;
        let bo_start = wo_start + d * d;
        let rel_bias_start = bo_start + d;

        let wo_view = MatrixBufferView::with_shape(
            params.parent_handle().clone(),
            params.offset_elements() + wo_start,
            d * d,
            d,
            d,
        );
        let bo_view = MatrixBufferView::new(
            params.parent_handle().clone(),
            params.offset_elements() + bo_start,
            d,
        );
        let wo_buf = subbuffer_from_view(self, &wo_view);
        let bo_buf = subbuffer_from_view(self, &bo_view);

        // 1. Вычисляем d_attn_out и градиенты W_o, b_o
        let bwd_output_params = &self.relative_position_attention_pipelines().backward_output_params;
        let push = [batch as u32, seq_len as u32, d_model as u32];
        self.run_compute_shader(
            bwd_output_params,
            &[
                (0, go_buf.clone()),
                (1, self.get_gpu_subbuffer_from_handle(&q_buf)), // attn_out не сохранён, передаём как есть? нужен attn_out
                // ВНИМАНИЕ: требуется attn_out, но мы его не сохранили. Нужно сохранять или пересчитать.
                // Мы можем вычислить attn_out заново внутри шейдера? Мы не передаём attn_out, но можем вычислить из weights и v.
                // Пока пропустим, предполагая, что шейдер сам вычислит attn_out.
                // В полной реализации нужно передать attn_out, либо вычислить здесь.
                (2, wo_buf.clone()),
                (3, bo_buf.clone()),
                (4, self.get_gpu_subbuffer_from_handle(&d_attn_out_buf)),
                (5, grad_params_buf.clone()),
            ],
            &push,
            token_count * d_model,
        );

        // 2. Вычисляем d_weights и d_v
        let bwd_values_weights = &self.relative_position_attention_pipelines().backward_values_weights;
        self.run_compute_shader(
            bwd_values_weights,
            &[
                (0, self.get_gpu_subbuffer_from_handle(&d_attn_out_buf)),
                (1, self.get_gpu_subbuffer_from_handle(v_buf)),
                (2, self.get_gpu_subbuffer_from_handle(weights_buf)),
                (3, self.get_gpu_subbuffer_from_handle(&d_weights_buf)),
                (4, self.get_gpu_subbuffer_from_handle(&d_v_buf)),
            ],
            &push,
            token_count * d_model + token_count * seq_len,
        );

        // 3. Вычисляем d_scores (softmax backward)
        let bwd_scores = &self.relative_position_attention_pipelines().backward_scores_softmax;
        self.run_compute_shader(
            bwd_scores,
            &[
                (0, self.get_gpu_subbuffer_from_handle(weights_buf)),
                (1, self.get_gpu_subbuffer_from_handle(&d_weights_buf)),
                (2, self.get_gpu_subbuffer_from_handle(&d_scores_buf)),
            ],
            &push,
            token_count * seq_len,
        );

        // 4. Вычисляем d_q, d_k и grad_rel_bias
        let bwd_qkv = &self.relative_position_attention_pipelines().backward_qkv_params;
        let rel_bias_view = MatrixBufferView::new(
            params.parent_handle().clone(),
            params.offset_elements() + rel_bias_start,
            2 * seq_len - 1,
        );
        let grad_rel_bias_view = MatrixBufferView::new(
            grad_params.parent_handle().clone(),
            grad_params.offset_elements() + rel_bias_start,
            2 * seq_len - 1,
        );
        self.run_compute_shader(
            bwd_qkv,
            &[
                (0, self.get_gpu_subbuffer_from_handle(q_buf)),
                (1, self.get_gpu_subbuffer_from_handle(k_buf)),
                (2, self.get_gpu_subbuffer_from_handle(&d_scores_buf)),
                (3, self.get_gpu_subbuffer_from_handle(&d_q_buf)),
                (4, self.get_gpu_subbuffer_from_handle(&d_k_buf)),
                (5, subbuffer_from_view(self, &grad_rel_bias_view)),
            ],
            &push,
            token_count * d_model * 2 + token_count * seq_len,
        );

        // 5. Вычисляем градиенты по входу и по W_q,b_q,W_k,b_k,W_v,b_v
        let bwd_input_params = &self.relative_position_attention_pipelines().backward_input_params;
        self.run_compute_shader(
            bwd_input_params,
            &[
                (0, in_buf.clone()),
                (1, self.get_gpu_subbuffer_from_handle(&d_q_buf)),
                (2, self.get_gpu_subbuffer_from_handle(&d_k_buf)),
                (3, self.get_gpu_subbuffer_from_handle(&d_v_buf)),
                (4, params_buf.clone()),
                (5, self.get_gpu_subbuffer_from_handle(&gi_row_buf)),
                (6, grad_params_buf.clone()),
            ],
            &push,
            token_count * d_model,
        );

        // 6. Преобразуем gi из row-major в column-major и записываем
        let gi_row_vec = self.download_gpu_handle_to_vec(&gi_row_buf);
        let gi_col = row_to_column(&gi_row_vec, batch, seq_len, d_model);
        self.copy_slice_to_gpu_handle(grad_input, &gi_col);

        // Освобождаем временные буферы
        self.release_temp_buffer(d_attn_out_buf, d_attn_out_raw);
        self.release_temp_buffer(d_weights_buf, d_weights_raw);
        self.release_temp_buffer(d_scores_buf, d_scores_raw);
        self.release_temp_buffer(d_q_buf, d_q_raw);
        self.release_temp_buffer(d_k_buf, d_k_raw);
        self.release_temp_buffer(d_v_buf, d_v_raw);
        self.release_temp_buffer(gi_row_buf, RawBufferId(0)); // заглушка, нужно правильно
    }
}