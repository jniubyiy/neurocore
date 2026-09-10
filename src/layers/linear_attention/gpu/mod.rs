// src/layers/linear_attention/gpu/mod.rs

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
    /// Прямой проход LinearAttention на GPU (column-major раскладка).
    ///
    /// Все промежуточные тензоры имеют форму `(batch, seq_len * d_model)` и
    /// хранятся в column-major. Матрица KV имеет форму `(d_model, d_model)` в
    /// column-major: `kv[i * d + l] = KV[l, i]`. Вектор Z имеет длину `d_model`.
    ///
    /// Параметры (Wq, bq, Wk, bk, Wv, bv, Wo, bo) передаются как единый view на
    /// плоский блок в `params`. Веса — row-major, смещения — линейные векторы.
    pub fn run_linear_attention_forward_buffered_handle_with_dims(
        &self,
        input: &MatrixBufferHandle,
        params: &MatrixBufferView,
        output: &MatrixBufferHandle,
        seq_len: usize,
        d_model: usize,
        q_raw: &MatrixBufferHandle,
        k_raw: &MatrixBufferHandle,
        v_raw: &MatrixBufferHandle,
        q_phi: &MatrixBufferHandle,
        k_phi: &MatrixBufferHandle,
        kv: &MatrixBufferHandle,
        z: &MatrixBufferHandle,
    ) {
        assert!(input.is_gpu(), "Input handle must be GPU");
        assert!(output.is_gpu(), "Output handle must be GPU");
        assert!(params.is_gpu(), "Params view must point to GPU buffer");

        let batch = input.rows();
        let features = seq_len * d_model;
        assert_eq!(input.cols(), features, "Input cols mismatch");
        assert_eq!(output.rows(), batch);
        assert_eq!(output.cols(), features, "Output cols mismatch");
        assert_eq!(
            params.len(),
            4 * (d_model * d_model + d_model),
            "Params length mismatch"
        );

        let token_count = batch * seq_len;
        let token_total = token_count * d_model;

        for buf in [q_raw, k_raw, v_raw, q_phi, k_phi] {
            assert_eq!(
                buf.rows() * buf.cols(),
                token_total,
                "Intermediate buffer size mismatch"
            );
        }
        assert_eq!(
            kv.rows() * kv.cols(),
            d_model * d_model,
            "kv size mismatch"
        );
        assert_eq!(z.rows() * z.cols(), d_model, "z size mismatch");

        let d = d_model;

        // Смещения параметров в плоском блоке:
        //   Wq: [0, d²)
        //   bq: [d², d² + d)
        //   Wk: [d² + d, 2d² + d)
        //   bk: [2d² + d, 2d² + 2d)
        //   Wv: [2d² + 2d, 3d² + 2d)
        //   bv: [3d² + 2d, 3d² + 3d)
        //   Wo: [3d² + 3d, 4d² + 3d)
        //   bo: [4d² + 3d, 4d² + 4d)
        let wq_off = 0usize;
        let bq_off = d * d;
        let wk_off = bq_off + d;
        let bk_off = wk_off + d * d;
        let wv_off = bk_off + d;
        let bv_off = wv_off + d * d;
        let wo_off = bv_off + d;
        let bo_off = wo_off + d * d;

        let parent = params.parent_handle().clone();
        let base = params.offset_elements();

        let wq_view = MatrixBufferView::with_shape(parent.clone(), base + wq_off, d * d, d, d);
        let bq_view = MatrixBufferView::new(parent.clone(), base + bq_off, d);
        let wk_view = MatrixBufferView::with_shape(parent.clone(), base + wk_off, d * d, d, d);
        let bk_view = MatrixBufferView::new(parent.clone(), base + bk_off, d);
        let wv_view = MatrixBufferView::with_shape(parent.clone(), base + wv_off, d * d, d, d);
        let bv_view = MatrixBufferView::new(parent.clone(), base + bv_off, d);
        let wo_view = MatrixBufferView::with_shape(parent.clone(), base + wo_off, d * d, d, d);
        let bo_view = MatrixBufferView::new(parent.clone(), base + bo_off, d);

        // 1. Per-token линейные преобразования Q, K, V.
        self.run_linear_per_token_forward_buffered_handle(
            input, &wq_view, &bq_view, q_raw, seq_len,
        );
        self.run_linear_per_token_forward_buffered_handle(
            input, &wk_view, &bk_view, k_raw, seq_len,
        );
        self.run_linear_per_token_forward_buffered_handle(
            input, &wv_view, &bv_view, v_raw, seq_len,
        );

        // 2. φ на q_raw и k_raw.
        let phi_pipeline = &self.linear_attention_pipelines().phi;
        let push_phi = [token_total as u32];

        self.run_compute_shader(
            phi_pipeline,
            &[
                (0, self.get_gpu_subbuffer_from_handle(q_raw)),
                (1, self.get_gpu_subbuffer_from_handle(q_phi)),
            ],
            &push_phi,
            token_total,
        );
        self.run_compute_shader(
            phi_pipeline,
            &[
                (0, self.get_gpu_subbuffer_from_handle(k_raw)),
                (1, self.get_gpu_subbuffer_from_handle(k_phi)),
            ],
            &push_phi,
            token_total,
        );

        // 3. Вычисление KV = K_phi^T · V и Z = K_phi^T · 1.
        let kvz_pipeline = &self.linear_attention_pipelines().compute_kvz;
        let push_kvz = [batch as u32, seq_len as u32, d_model as u32];
        self.run_compute_shader(
            kvz_pipeline,
            &[
                (0, self.get_gpu_subbuffer_from_handle(k_phi)),
                (1, self.get_gpu_subbuffer_from_handle(v_raw)),
                (2, self.get_gpu_subbuffer_from_handle(kv)),
                (3, self.get_gpu_subbuffer_from_handle(z)),
            ],
            &push_kvz,
            d_model * d_model + d_model,
        );

        // 4. Основной forward: y = b_o + φ(Q) · KV · W_o^T / (φ(Q) · Z + eps).
        let fwd_pipeline = &self.linear_attention_pipelines().forward;
        let push_fwd = [batch as u32, seq_len as u32, d_model as u32];
        self.run_compute_shader(
            fwd_pipeline,
            &[
                (0, self.get_gpu_subbuffer_from_handle(q_phi)),
                (1, self.get_gpu_subbuffer_from_handle(kv)),
                (2, self.get_gpu_subbuffer_from_handle(z)),
                (3, subbuffer_from_view(self, &wo_view)),
                (4, subbuffer_from_view(self, &bo_view)),
                (5, self.get_gpu_subbuffer_from_handle(output)),
            ],
            &push_fwd,
            token_total,
        );
    }

    /// Обратный проход LinearAttention на GPU (column-major раскладка).
    ///
    /// Принимает сохранённые промежуточные буферы с forward. Градиенты по
    /// параметрам записываются в `grad_params` через атомарное накопление.
    pub fn run_linear_attention_backward_buffered_handle_with_dims(
        &self,
        input: &MatrixBufferHandle,
        grad_out: &MatrixBufferHandle,
        params: &MatrixBufferView,
        grad_input: &MatrixBufferHandle,
        grad_params: &MatrixBufferView,
        seq_len: usize,
        d_model: usize,
        q_raw: &MatrixBufferHandle,
        k_raw: &MatrixBufferHandle,
        v_raw: &MatrixBufferHandle,
        q_phi: &MatrixBufferHandle,
        k_phi: &MatrixBufferHandle,
        kv: &MatrixBufferHandle,
        z: &MatrixBufferHandle,
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
        assert_eq!(
            params.len(),
            4 * (d_model * d_model + d_model),
            "Params length mismatch"
        );
        assert_eq!(
            grad_params.len(),
            4 * (d_model * d_model + d_model),
            "grad_params length mismatch"
        );

        let token_count = batch * seq_len;
        let token_total = token_count * d_model;
        let d = d_model;

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

        // 2. Временные буферы для обратного прохода.
        let (d_attn_out_buf, d_attn_out_raw) = self.acquire_temp_buffer(token_total);
        let (d_q_phi_buf, d_q_phi_raw) = self.acquire_temp_buffer(token_total);
        let (d_kv_buf, d_kv_raw) = self.acquire_temp_buffer(d * d);
        let (d_z_buf, d_z_raw) = self.acquire_temp_buffer(d);

        // 3. Обнуляем d_kv и d_z (используются с атомарным накоплением).
        let zero_kv = self.upload_vec_to_gpu_handle(&vec![0.0f32; d * d], d * d, 1);
        self.copy_buffer_sync(
            self.get_gpu_subbuffer_from_handle(&zero_kv),
            d_kv_buf.clone(),
        );
        let zero_z = self.upload_vec_to_gpu_handle(&vec![0.0f32; d], d, 1);
        self.copy_buffer_sync(
            self.get_gpu_subbuffer_from_handle(&zero_z),
            d_z_buf.clone(),
        );

        // 4. Извлекаем view на W_o (для первого этапа backward).
        let wo_off = 3 * d * d + 3 * d;
        let wo_view = MatrixBufferView::with_shape(
            params.parent_handle().clone(),
            params.offset_elements() + wo_off,
            d * d,
            d,
            d,
        );
        let wo_buf = subbuffer_from_view(self, &wo_view);

        // 5. Первый этап: bwd_main — d_attn_out, d_q_phi, d_kv, d_z.
        let bwd_main_pipeline = &self.linear_attention_pipelines().backward_main;
        let push_bwd = [batch as u32, seq_len as u32, d_model as u32];
        self.run_compute_shader(
            bwd_main_pipeline,
            &[
                (0, self.get_gpu_subbuffer_from_handle(grad_out)),
                (1, wo_buf),
                (2, self.get_gpu_subbuffer_from_handle(q_phi)),
                (3, self.get_gpu_subbuffer_from_handle(kv)),
                (4, self.get_gpu_subbuffer_from_handle(z)),
                (5, d_attn_out_buf.clone()),
                (6, d_q_phi_buf.clone()),
                (7, d_kv_buf.clone()),
                (8, d_z_buf.clone()),
            ],
            &push_bwd,
            token_count,
        );

        // 6. Второй этап: bwd_params — gi и градиенты параметров.
        let bwd_params_pipeline = &self.linear_attention_pipelines().backward_params;
        self.run_compute_shader(
            bwd_params_pipeline,
            &[
                (0, self.get_gpu_subbuffer_from_handle(input)),
                (1, self.get_gpu_subbuffer_from_handle(grad_out)),
                (2, subbuffer_from_view(self, params)),
                (3, self.get_gpu_subbuffer_from_handle(q_phi)),
                (4, self.get_gpu_subbuffer_from_handle(k_phi)),
                (5, self.get_gpu_subbuffer_from_handle(v_raw)),
                (6, self.get_gpu_subbuffer_from_handle(q_raw)),
                (7, self.get_gpu_subbuffer_from_handle(k_raw)),
                (8, d_q_phi_buf.clone()),
                (9, d_kv_buf.clone()),
                (10, d_z_buf.clone()),
                (11, self.get_gpu_subbuffer_from_handle(grad_input)),
                (12, subbuffer_from_view(self, grad_params)),
                (13, self.get_gpu_subbuffer_from_handle(kv)),
                (14, self.get_gpu_subbuffer_from_handle(z)),
            ],
            &push_bwd,
            token_count,
        );

        // 7. Освобождение временных буферов.
        self.release_temp_buffer(d_attn_out_buf, d_attn_out_raw);
        self.release_temp_buffer(d_q_phi_buf, d_q_phi_raw);
        self.release_temp_buffer(d_kv_buf, d_kv_raw);
        self.release_temp_buffer(d_z_buf, d_z_raw);
    }

    /// Вспомогательный метод: per-token линейное преобразование.
    ///
    /// Применяет матрицу весов `(out_features, in_features)` row-major и смещение
    /// `(out_features,)` к каждому токену входного тензора `(batch, seq_len * in_features)`
    /// в column-major. Результат — `(batch, seq_len * out_features)` в column-major.
    fn run_linear_per_token_forward_buffered_handle(
        &self,
        input: &MatrixBufferHandle,
        weight: &MatrixBufferView,
        bias: &MatrixBufferView,
        output: &MatrixBufferHandle,
        seq_len: usize,
    ) {
        assert!(input.is_gpu(), "Input handle must be GPU");
        assert!(output.is_gpu(), "Output handle must be GPU");
        assert!(weight.is_gpu(), "Weight view must point to GPU buffer");
        assert!(bias.is_gpu(), "Bias view must point to GPU buffer");
        assert!(seq_len > 0, "seq_len must be positive");

        let batch = input.rows();
        let in_features = weight.cols();
        let out_features = weight.rows();

        assert_eq!(
            input.cols(),
            seq_len * in_features,
            "Input cols mismatch for per-token linear"
        );
        assert_eq!(
            output.rows(),
            batch,
            "Output rows mismatch for per-token linear"
        );
        assert_eq!(
            output.cols(),
            seq_len * out_features,
            "Output cols mismatch for per-token linear"
        );
        assert_eq!(bias.len(), out_features, "Bias length mismatch");

        let in_buf = self.get_gpu_subbuffer_from_handle(input);
        let w_buf = subbuffer_from_view(self, weight);
        let b_buf = subbuffer_from_view(self, bias);
        let out_buf = self.get_gpu_subbuffer_from_handle(output);

        let pipeline = &self.linear_attention_pipelines().linear_per_token;
        let push = [
            batch as u32,
            seq_len as u32,
            in_features as u32,
            out_features as u32,
        ];
        let total = batch * seq_len * out_features;

        self.run_compute_shader(
            pipeline,
            &[
                (0, in_buf),
                (1, w_buf),
                (2, b_buf),
                (3, out_buf),
            ],
            &push,
            total,
        );
    }
}