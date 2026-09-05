// src/layers/ind_rnn/gpu/mod.rs

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
    /// Прямой проход IndRNN на GPU.
    ///
    /// Выполняет последовательные шаги по времени `t` от 0 до `seq_len-1`.
    /// Скрытые состояния сохраняются в `h_all` для использования в обратном проходе.
    ///
    /// # Аргументы
    /// * `input` – вход `(batch, seq_len * input_dim)` column-major.
    /// * `params` – view на блок параметров `[W; u; b]` размером `input_dim² + 2*input_dim`.
    /// * `output` – выход `(batch, seq_len * input_dim)` column-major.
    /// * `seq_len` – длина последовательности.
    /// * `input_dim` – размерность признаков.
    /// * `h_all` – GPU-буфер размера `batch * seq_len * input_dim` (row-major по `(r,t,i)`).
    ///   Должен быть выделен вызывающим кодом, чтобы пережить вызов и использоваться в backward.
    pub fn run_ind_rnn_forward_buffered_handle(
        &self,
        input: &MatrixBufferHandle,
        params: &MatrixBufferView,
        output: &MatrixBufferHandle,
        seq_len: usize,
        input_dim: usize,
        h_all: &MatrixBufferHandle,
    ) {
        assert!(input.is_gpu(), "Input handle must be GPU");
        assert!(output.is_gpu(), "Output handle must be GPU");
        assert!(params.is_gpu(), "Params view must point to GPU buffer");

        let batch = input.rows();
        let features = input_dim;
        assert_eq!(input.cols(), seq_len * features, "Input cols mismatch");
        assert_eq!(output.rows(), batch);
        assert_eq!(output.cols(), seq_len * features, "Output cols mismatch");
        assert_eq!(h_all.rows() * h_all.cols(), batch * seq_len * features, "h_all size mismatch");

        let param_len = input_dim * input_dim + 2 * input_dim;
        assert_eq!(params.len(), param_len, "Params length mismatch");

        // Создаём view для W, u, b
        let d = input_dim;
        let w_start = 0;
        let u_start = w_start + d * d;
        let b_start = u_start + d;

        let w_view = MatrixBufferView::with_shape(
            params.parent_handle().clone(),
            params.offset_elements() + w_start,
            d * d,
            d,
            d,
        );
        let u_view = MatrixBufferView::new(
            params.parent_handle().clone(),
            params.offset_elements() + u_start,
            d,
        );
        let b_view = MatrixBufferView::new(
            params.parent_handle().clone(),
            params.offset_elements() + b_start,
            d,
        );

        let in_buf = self.get_gpu_subbuffer_from_handle(input);
        let out_buf = self.get_gpu_subbuffer_from_handle(output);
        let h_buf = self.get_gpu_subbuffer_from_handle(h_all);
        let w_buf = subbuffer_from_view(self, &w_view);
        let u_buf = subbuffer_from_view(self, &u_view);
        let b_buf = subbuffer_from_view(self, &b_view);

        let pipeline = &self.ind_rnn_pipelines().forward_step;
        let total_elems = batch * features;

        for t in 0..seq_len {
            let push = [batch as u32, features as u32, seq_len as u32, t as u32];
            self.run_compute_shader(
                pipeline,
                &[
                    (0, in_buf.clone()),
                    (1, w_buf.clone()),
                    (2, u_buf.clone()),
                    (3, b_buf.clone()),
                    (4, h_buf.clone()),
                    (5, out_buf.clone()),
                ],
                &push,
                total_elems,
            );
        }
    }

    /// Обратный проход IndRNN на GPU.
    ///
    /// Выполняет шаги в обратном порядке от `seq_len-1` до 0.
    /// Градиенты параметров атомарно накапливаются в `grad_params`.
    /// Градиент по входу накапливается в `grad_input` (также атомарно).
    ///
    /// # Аргументы
    /// * `input` – вход `(batch, seq_len * input_dim)` column-major.
    /// * `grad_out` – градиент по выходу (тот же формат).
    /// * `params` – view на параметры `[W; u; b]`.
    /// * `grad_input` – GPU-буфер для градиента по входу (будет обнулён и заполнен).
    /// * `grad_params` – view на градиенты параметров `[grad_W; grad_u; grad_b]`.
    /// * `seq_len` – длина последовательности.
    /// * `input_dim` – размерность признаков.
    /// * `h_all` – GPU-буфер скрытых состояний из forward.
    pub fn run_ind_rnn_backward_buffered_handle(
        &self,
        input: &MatrixBufferHandle,
        grad_out: &MatrixBufferHandle,
        params: &MatrixBufferView,
        grad_input: &MatrixBufferHandle,
        grad_params: &MatrixBufferView,
        seq_len: usize,
        input_dim: usize,
        h_all: &MatrixBufferHandle,
    ) {
        assert!(input.is_gpu(), "Input handle must be GPU");
        assert!(grad_out.is_gpu(), "grad_out handle must be GPU");
        assert!(grad_input.is_gpu(), "grad_input handle must be GPU");
        assert!(grad_params.is_gpu(), "grad_params view must point to GPU buffer");
        assert!(params.is_gpu(), "Params view must point to GPU buffer");

        let batch = input.rows();
        let features = input_dim;
        assert_eq!(input.cols(), seq_len * features, "Input cols mismatch");
        assert_eq!(grad_out.rows(), batch);
        assert_eq!(grad_out.cols(), seq_len * features, "grad_out cols mismatch");
        assert_eq!(grad_input.rows(), batch);
        assert_eq!(grad_input.cols(), seq_len * features, "grad_input cols mismatch");

        let param_len = input_dim * input_dim + 2 * input_dim;
        assert_eq!(params.len(), param_len, "Params length mismatch");
        assert_eq!(grad_params.len(), param_len, "Grad params length mismatch");
        assert_eq!(h_all.rows() * h_all.cols(), batch * seq_len * features, "h_all size mismatch");

        // Обнуляем градиенты по параметрам
        let zero_params = self.upload_vec_to_gpu_handle(
            &vec![0.0f32; grad_params.len()],
            grad_params.len(),
            1,
        );
        self.copy_gpu_handle_region(
            &zero_params,
            grad_params.parent_handle(),
            0,
            grad_params.offset_elements(),
            grad_params.len(),
        );

        // Обнуляем градиент по входу
        self.fill_gpu_handle(grad_input, 0.0);

        // Создаём view для W, u, b и соответствующих градиентов
        let d = input_dim;
        let w_start = 0;
        let u_start = w_start + d * d;
        let b_start = u_start + d;

        let w_view = MatrixBufferView::with_shape(
            params.parent_handle().clone(),
            params.offset_elements() + w_start,
            d * d,
            d,
            d,
        );
        let u_view = MatrixBufferView::new(
            params.parent_handle().clone(),
            params.offset_elements() + u_start,
            d,
        );
        let b_view = MatrixBufferView::new(
            params.parent_handle().clone(),
            params.offset_elements() + b_start,
            d,
        );

        let gw_view = MatrixBufferView::with_shape(
            grad_params.parent_handle().clone(),
            grad_params.offset_elements() + w_start,
            d * d,
            d,
            d,
        );
        let gu_view = MatrixBufferView::new(
            grad_params.parent_handle().clone(),
            grad_params.offset_elements() + u_start,
            d,
        );
        let gb_view = MatrixBufferView::new(
            grad_params.parent_handle().clone(),
            grad_params.offset_elements() + b_start,
            d,
        );

        // Subbuffer'ы
        let in_buf = self.get_gpu_subbuffer_from_handle(input);
        let go_buf = self.get_gpu_subbuffer_from_handle(grad_out);
        let h_buf = self.get_gpu_subbuffer_from_handle(h_all);
        let w_buf = subbuffer_from_view(self, &w_view);
        let u_buf = subbuffer_from_view(self, &u_view);
        let b_buf = subbuffer_from_view(self, &b_view);
        let gi_buf = self.get_gpu_subbuffer_from_handle(grad_input);
        let gw_buf = subbuffer_from_view(self, &gw_view);
        let gu_buf = subbuffer_from_view(self, &gu_view);
        let gb_buf = subbuffer_from_view(self, &gb_view);

        // Временные буферы для delta_next (передаются между шагами)
        let (delta_a_buf, delta_a_raw) = self.acquire_temp_buffer(batch * features);
        let (delta_b_buf, delta_b_raw) = self.acquire_temp_buffer(batch * features);
        // Инициализируем первый вход нулями (для t=seq_len-1)
        self.fill_gpu_handle(&MatrixBufferHandle::from_existing_sub?);

        // Здесь нужно заполнить delta_a_buf нулями. Поскольку delta_a_buf это Subbuffer, а не handle,
        // мы используем метод fill_gpu_handle на временном handle.
        // Проще: создадим два MatrixBufferHandle, чтобы использовать fill_gpu_handle.
        // Но acquire_temp_buffer возвращает (Subbuffer, RawBufferId). Чтобы обнулить, можно
        // использовать self.run_compute_shader с шейдером fill? Или использовать copy из нулевого буфера.
        // Для простоты: выделим временные MatrixBufferHandle вместо raw Subbuffer.
        // Но у нас есть метод acquire_temp_buffer, возвращающий Subbuffer. Мы можем обнулить его,
        // используя zero_handle и copy_buffer_sync? Но copy_buffer_sync работает на Subbuffer.
        // Мы можем создать нулевой Subbuffer, загрузив нули через upload_to_temp_buffer,
        // затем скопировать в delta_a_buf. Или использовать run_compute_shader с простым шейдером fill.
        // Чтобы не усложнять, мы можем обнулить delta_a_buf, запустив fill через вспомогательный
        // шейдер, но такого нет. Вместо этого мы можем создать delta_a_buf, используя acquire_temp_buffer,
        // и затем запустить fill_gpu_handle на соответствующем MatrixBufferHandle? Невозможно.
        // Поэтому примем, что мы выделяем MatrixBufferHandle для delta-буферов через allocate_gpu_matrix_handle,
        // чтобы использовать fill_gpu_handle. Это не идеально, но допустимо для первой реализации.
        // В реальности можно создать нулевой буфер и скопировать.

        // Для простоты я буду использовать MatrixBufferHandle для delta-буферов.
        let delta_in_handle = self.allocate_gpu_matrix_handle(batch, features);
        let delta_out_handle = self.allocate_gpu_matrix_handle(batch, features);
        self.fill_gpu_handle(&delta_in_handle, 0.0);

        let pipeline = &self.ind_rnn_pipelines().backward_step;
        let total_elems = batch * features;

        for t in (0..seq_len).rev() {
            let push = [batch as u32, features as u32, seq_len as u32, t as u32];

            // Передаём delta_in_handle как входной, delta_out_handle как выходной.
            // После шага меняем местами для следующего шага.
            self.run_compute_shader(
                pipeline,
                &[
                    (0, in_buf.clone()),
                    (1, w_buf.clone()),
                    (2, u_buf.clone()),
                    (3, b_buf.clone()),
                    (4, go_buf.clone()),
                    (5, h_buf.clone()),
                    (6, self.get_gpu_subbuffer_from_handle(&delta_in_handle)),
                    (7, gw_buf.clone()),
                    (8, gu_buf.clone()),
                    (9, gb_buf.clone()),
                    (10, gi_buf.clone()),
                    (11, self.get_gpu_subbuffer_from_handle(&delta_out_handle)),
                ],
                &push,
                total_elems,
            );

            // Обмениваем буферы для следующего шага
            std::mem::swap(&mut delta_in_handle, &mut delta_out_handle);
            // Но поскольку мы поменяли handle'ы, то для следующего шага входной delta_next будет
            // тот, что был выходным на этом шаге. Это правильно.
        }

        // Освобождаем временные матричные handle'ы
        drop(delta_in_handle);
        drop(delta_out_handle);
    }
}