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
    /// Прямой проход LinearAttention на GPU.
    ///
    /// Параметры слоя (все матрицы и смещения) передаются как `params_view`.
    /// Вход и выход — GPU-дескрипторы. Внутри выделяются временные буферы
    /// для промежуточных вычислений, но итоговые промежуточные результаты
    /// (q_raw, k_raw, v_raw, q_phi, k_phi, kv, z) сохраняются для обратного прохода.
    ///
    /// # Аргументы
    /// * `input` – вход `(batch, seq_len * d_model)` (column-major, GPU).
    /// * `params` – view на полный блок параметров `4*(d_model² + d_model)`.
    /// * `output` – выход `(batch, seq_len * d_model)`.
    /// * `seq_len`, `d_model` – размеры последовательности и модели.
    /// * `q_raw`, `k_raw`, `v_raw` – буферы для сохранения линейных преобразований
    ///   (до применения φ). Размер каждого: `(batch*seq_len) x d_model`, row-major.
    /// * `q_phi`, `k_phi` – буферы после применения φ (row-major).
    /// * `kv`, `z` – буферы для сохранения матрицы KᵀV и вектора Kᵀ1.
    ///   `kv` имеет размер `d_model x d_model`, `z` – `d_model`.
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
        assert_eq!(params.len(), 4 * (d_model * d_model + d_model), "Params length mismatch");

        // Проверяем размеры промежуточных буферов
        let token_count = batch * seq_len;
        for buf in [q_raw, k_raw, v_raw, q_phi, k_phi] {
            assert_eq!(
                buf.rows() * buf.cols(),
                token_count * d_model,
                "Intermediate buffer size mismatch"
            );
        }
        assert_eq!(kv.rows() * kv.cols(), d_model * d_model, "kv size mismatch");
        assert_eq!(z.rows() * z.cols(), d_model, "z size mismatch");

        // Создаём views для всех параметров
        let d = d_model;
        let wq_start = 0usize;
        let bq_start = wq_start + d * d;
        let wk_start = bq_start + d;
        let bk_start = wk_start + d * d;
        let wv_start = bk_start + d;
        let bv_start = wv_start + d * d;
        let wo_start = bv_start + d;
        let bo_start = wo_start + d * d;

        let wq_view = MatrixBufferView::with_shape(
            params.parent_handle().clone(),
            params.offset_elements() + wq_start,
            d * d,
            d,
            d,
        );
        let bq_view = MatrixBufferView::new(
            params.parent_handle().clone(),
            params.offset_elements() + bq_start,
            d,
        );
        let wk_view = MatrixBufferView::with_shape(
            params.parent_handle().clone(),
            params.offset_elements() + wk_start,
            d * d,
            d,
            d,
        );
        let bk_view = MatrixBufferView::new(
            params.parent_handle().clone(),
            params.offset_elements() + bk_start,
            d,
        );
        let wv_view = MatrixBufferView::with_shape(
            params.parent_handle().clone(),
            params.offset_elements() + wv_start,
            d * d,
            d,
            d,
        );
        let bv_view = MatrixBufferView::new(
            params.parent_handle().clone(),
            params.offset_elements() + bv_start,
            d,
        );
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

        // 1. Линейные преобразования для Q, K, V
        self.run_linear_forward_buffered_handle(input, &wq_view, &bq_view, q_raw);
        self.run_linear_forward_buffered_handle(input, &wk_view, &bk_view, k_raw);
        self.run_linear_forward_buffered_handle(input, &wv_view, &bv_view, v_raw);

        // 2. Применяем φ(x) = ELU(x)+1 к q_raw и k_raw
        let phi_pipeline = &self.linear_attention_pipelines().phi;
        let push_phi = [(token_count * d_model) as u32];
        let q_raw_buf = self.get_gpu_subbuffer_from_handle(q_raw);
        let k_raw_buf = self.get_gpu_subbuffer_from_handle(k_raw);
        let q_phi_buf = self.get_gpu_subbuffer_from_handle(q_phi);
        let k_phi_buf = self.get_gpu_subbuffer_from_handle(k_phi);

        self.run_compute_shader(
            phi_pipeline,
            &[(0, q_raw_buf), (1, q_phi_buf)],
            &push_phi,
            token_count * d_model,
        );
        self.run_compute_shader(
            phi_pipeline,
            &[(0, k_raw_buf), (1, k_phi_buf)],
            &push_phi,
            token_count * d_model,
        );

        // 3. Вычисляем KV и Z
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

        // 4. Основной forward (выход Y = φ(Q)(KᵀV)/(φ(Q)Kᵀ1) W_o + b_o)
        let fwd_pipeline = &self.linear_attention_pipelines().forward;
        let push_fwd = [batch as u32, seq_len as u32, d_model as u32];
        let total_out = batch * seq_len * d_model;
        self.run_compute_shader(
            fwd_pipeline,
            &[
                (0, self.get_gpu_subbuffer_from_handle(q_phi)),
                (1, self.get_gpu_subbuffer_from_handle(k_phi)), // не используется в этом шейдере
                (2, self.get_gpu_subbuffer_from_handle(v_raw)), // не используется
                (3, self.get_gpu_subbuffer_from_handle(kv)),
                (4, self.get_gpu_subbuffer_from_handle(z)),
                (5, subbuffer_from_view(self, &wo_view)),
                (6, subbuffer_from_view(self, &bo_view)),
                (7, self.get_gpu_subbuffer_from_handle(output)),
            ],
            &push_fwd,
            total_out,
        );
    }

    /// Обратный проход LinearAttention на GPU.
    ///
    /// Принимает сохранённые промежуточные буферы с forward.
    /// Градиенты по параметрам записываются в `grad_params` (view на полный блок).
    /// Вход/выходные градиенты — GPU-дескрипторы.
    ///
    /// # Аргументы
    /// * `input` – вход `(batch, seq_len * d_model)`.
    /// * `grad_out` – градиент по выходу.
    /// * `params` – view на параметры.
    /// * `grad_input` – буфер для градиента по входу.
    /// * `grad_params` – view на градиенты параметров.
    /// * `seq_len`, `d_model` – размеры.
    /// * Промежуточные буферы из forward: q_raw, k_raw, v_raw, q_phi, k_phi, kv, z.
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
        assert_eq!(params.len(), 4 * (d_model * d_model + d_model), "Params length mismatch");
        assert_eq!(grad_params.len(), 4 * (d_model * d_model + d_model), "grad_params length mismatch");

        let token_count = batch * seq_len;
        let d = d_model;

        // Обнуляем градиенты параметров
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

        // Выделяем временные буферы для обратного прохода
        let (d_attn_out_buf, d_attn_out_raw) = self.acquire_temp_buffer(token_count * d);
        let (d_q_phi_buf, d_q_phi_raw) = self.acquire_temp_buffer(token_count * d);
        let (d_kv_buf, d_kv_raw) = self.acquire_temp_buffer(d * d);
        let (d_z_buf, d_z_raw) = self.acquire_temp_buffer(d);

        // Обнуляем d_kv и d_z перед накоплением (atomic)
        let zero_kvz = self.upload_vec_to_gpu_handle(&vec![0.0f32; d * d], d * d, 1);
        self.copy_buffer_sync(
            self.get_gpu_subbuffer_from_handle(&zero_kvz),
            d_kv_buf.clone(),
        );
        let zero_z = self.upload_vec_to_gpu_handle(&vec![0.0f32; d], d, 1);
        self.copy_buffer_sync(
            self.get_gpu_subbuffer_from_handle(&zero_z),
            d_z_buf.clone(),
        );

        // Получаем subbuffer'ы для параметров
        let wo_view = MatrixBufferView::with_shape(
            params.parent_handle().clone(),
            params.offset_elements() + (4 * (d * d + d) - d * d - d),
            d * d,
            d,
            d,
        );
        let wo_buf = subbuffer_from_view(self, &wo_view);

        // 1. Backward main: вычисляет d_attn_out, d_q_phi, d_kv, d_z
        let bwd_main_pipeline = &self.linear_attention_pipelines().backward_main;
        let push_bwd_main = [batch as u32, seq_len as u32, d_model as u32];
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
            &push_bwd_main,
            token_count,
        );

        // 2. Backward params: вычисляет gi и градиенты по параметрам, включая W_o/b_o
        let bwd_params_pipeline = &self.linear_attention_pipelines().backward_params;
        let push_bwd_params = [batch as u32, seq_len as u32, d_model as u32];
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
                (8, d_q_phi_buf.clone()), // d_q_phi
                (9, d_kv_buf.clone()),   // d_kv
                (10, d_z_buf.clone()),   // d_z
                (11, self.get_gpu_subbuffer_from_handle(grad_input)), // gi
                (12, subbuffer_from_view(self, grad_params)),        // grad_params
                // Дополнительные буферы kv и z для вычисления attn_out внутри шейдера
                (13, self.get_gpu_subbuffer_from_handle(kv)),
                (14, self.get_gpu_subbuffer_from_handle(z)),
            ],
            &push_bwd_params,
            batch * seq_len * d_model,
        );

        // Освобождаем временные буферы
        self.release_temp_buffer(d_attn_out_buf, d_attn_out_raw);
        self.release_temp_buffer(d_q_phi_buf, d_q_phi_raw);
        self.release_temp_buffer(d_kv_buf, d_kv_raw);
        self.release_temp_buffer(d_z_buf, d_z_raw);
    }
}