// src/layers/mamba/gpu/mod.rs

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
    /// Прямой проход Mamba на GPU.
    ///
    /// Выполняет дискретизацию A и B, затем последовательно проходит все шаги времени.
    /// Скрытые состояния сохраняются в `h_all` для последующего обратного прохода.
    ///
    /// # Аргументы
    /// * `input` – вход `(batch, seq_len * input_dim)`, column-major.
    /// * `params` – view на блок параметров `[A; B; C; D; delta]` размером `n*n + n*d + d*n + 2`.
    /// * `output` – выход `(batch, seq_len * input_dim)`, column-major.
    /// * `seq_len` – длина последовательности.
    /// * `input_dim` – размерность входа/выхода (`d`).
    /// * `state_dim` – размерность скрытого состояния (`n`).
    /// * `h_all` – GPU-буфер размера `batch * seq_len * state_dim` (row-major по `(r,t,i)`),
    ///   выделяется вызывающим кодом и используется в backward.
    pub fn run_mamba_forward_buffered_handle(
        &self,
        input: &MatrixBufferHandle,
        params: &MatrixBufferView,
        output: &MatrixBufferHandle,
        seq_len: usize,
        input_dim: usize,
        state_dim: usize,
        h_all: &MatrixBufferHandle,
    ) {
        assert!(input.is_gpu(), "Input handle must be GPU");
        assert!(output.is_gpu(), "Output handle must be GPU");
        assert!(params.is_gpu(), "Params view must point to GPU buffer");

        let batch = input.rows();
        let d = input_dim;
        let n = state_dim;
        assert_eq!(input.cols(), seq_len * d, "Input cols mismatch");
        assert_eq!(output.rows(), batch);
        assert_eq!(output.cols(), seq_len * d, "Output cols mismatch");
        assert_eq!(h_all.rows() * h_all.cols(), batch * seq_len * n, "h_all size mismatch");

        let param_len = n * n + n * d + d * n + 2;
        assert_eq!(params.len(), param_len, "Params length mismatch");

        // Создаём view для параметров
        let a_start = 0usize;
        let b_start = a_start + n * n;
        let c_start = b_start + n * d;
        let d_idx = c_start + d * n;
        let delta_idx = d_idx + 1;

        let a_view = MatrixBufferView::with_shape(
            params.parent_handle().clone(),
            params.offset_elements() + a_start,
            n * n,
            n,
            n,
        );
        let b_view = MatrixBufferView::with_shape(
            params.parent_handle().clone(),
            params.offset_elements() + b_start,
            n * d,
            n,
            d,
        );
        let c_view = MatrixBufferView::with_shape(
            params.parent_handle().clone(),
            params.offset_elements() + c_start,
            d * n,
            d,
            n,
        );
        let d_view = MatrixBufferView::new(
            params.parent_handle().clone(),
            params.offset_elements() + d_idx,
            1,
        );
        let delta_view = MatrixBufferView::new(
            params.parent_handle().clone(),
            params.offset_elements() + delta_idx,
            1,
        );

        // Получаем значения D и delta (скаляры) для push-констант и дискретизации
        // Проще загрузить их на CPU через download, но можно и через отдельные view.
        // Для простоты скачаем D и delta через CPU, так как они скаляры.
        let d_val = {
            let handle = self.download_gpu_handle_to_cpu_handle(d_view.parent_handle());
            let guard = handle.read();
            let slice = guard.as_slice().unwrap();
            slice[d_view.offset_elements()]
        };
        let delta_val = {
            let handle = self.download_gpu_handle_to_cpu_handle(delta_view.parent_handle());
            let guard = handle.read();
            let slice = guard.as_slice().unwrap();
            slice[delta_view.offset_elements()]
        };

        // Выделяем временные буферы A_bar и B_bar
        let (a_bar_buf, a_bar_raw) = self.acquire_temp_buffer(n * n);
        let (b_bar_buf, b_bar_raw) = self.acquire_temp_buffer(n * d);

        // Subbuffer'ы
        let in_buf = self.get_gpu_subbuffer_from_handle(input);
        let out_buf = self.get_gpu_subbuffer_from_handle(output);
        let h_buf = self.get_gpu_subbuffer_from_handle(h_all);
        let a_buf = subbuffer_from_view(self, &a_view);
        let b_buf = subbuffer_from_view(self, &b_view);
        let c_buf = subbuffer_from_view(self, &c_view);

        // Запускаем дискретизацию
        let discretize_pipeline = &self.mamba_pipelines().discretize;
        let push_disc = [n as u32, d as u32, delta_val.to_bits(), 0u32];
        let total_disc = n * n + n * d;
        self.run_compute_shader(
            discretize_pipeline,
            &[
                (0, a_buf.clone()),
                (1, b_buf.clone()),
                (2, a_bar_buf.clone()),
                (3, b_bar_buf.clone()),
            ],
            &push_disc,
            total_disc,
        );

        // Последовательные шаги
        let fwd_pipeline = &self.mamba_pipelines().forward_step;

        for t in 0..seq_len {
            // Фаза 0: вычисление h_t
            let push_h = [
                batch as u32,
                d as u32,
                n as u32,
                seq_len as u32,
                t as u32,
                0u32,
            ];
            self.run_compute_shader(
                fwd_pipeline,
                &[
                    (0, in_buf.clone()),
                    (1, a_bar_buf.clone()),
                    (2, b_bar_buf.clone()),
                    (3, c_buf.clone()),
                    (4, d_val.to_bits() as f32), // D не используется в фазе 0, но шейдер требует буфер? Нет, D передаётся как буфер, но в фазе 0 не используется. В нашем шейдере binding 4 — это D как буфер? Мы сделали binding 4 как DBuf с одним float? В шейдере mamba_fwd_step.comp binding 4 объявлен как `DBuf { float D; }`, то есть это storage buffer с одним элементом. Поэтому нужно передавать Subbuffer на D_view, а не push-константу. Но у нас есть d_view, который является view на один элемент. Мы можем получить Subbuffer для d_view через subbuffer_from_view. Исправим: в шейдере мы ожидаем буфер, поэтому в Rust коде нужно передать subbuffer для D. Сейчас d_val мы скачали, но для шейдера нужен именно буфер. Поэтому передадим subbuffer_from_view(self, &d_view).
                    // Передаём D как буфер
                    (4, subbuffer_from_view(self, &d_view)),
                    (5, h_buf.clone()),
                    (6, out_buf.clone()),
                ],
                &push_h,
                batch * n,
            );

            // Фаза 1: вычисление y_t
            let push_y = [
                batch as u32,
                d as u32,
                n as u32,
                seq_len as u32,
                t as u32,
                1u32,
            ];
            self.run_compute_shader(
                fwd_pipeline,
                &[
                    (0, in_buf.clone()),
                    (1, a_bar_buf.clone()),
                    (2, b_bar_buf.clone()),
                    (3, c_buf.clone()),
                    (4, subbuffer_from_view(self, &d_view)),
                    (5, h_buf.clone()),
                    (6, out_buf.clone()),
                ],
                &push_y,
                batch * d,
            );
        }

        // Освобождаем временные буферы
        self.release_temp_buffer(a_bar_buf, a_bar_raw);
        self.release_temp_buffer(b_bar_buf, b_bar_raw);
    }

    /// Обратный проход Mamba на GPU.
    ///
    /// Выполняет обратные шаги по времени, накапливая градиенты по A_bar, B_bar, C, D
    /// и входу. Затем преобразует градиенты A_bar, B_bar в градиенты A, B, delta.
    ///
    /// # Аргументы
    /// * `input` – вход `(batch, seq_len * input_dim)`, column-major.
    /// * `grad_out` – градиент по выходу (тот же формат).
    /// * `params` – view на параметры `[A; B; C; D; delta]`.
    /// * `grad_input` – GPU-буфер для градиента по входу (будет заполнен).
    /// * `grad_params` – view на градиенты параметров (той же длины, что и параметры).
    /// * `seq_len`, `input_dim`, `state_dim` – размеры.
    /// * `h_all` – GPU-буфер скрытых состояний из forward.
    pub fn run_mamba_backward_buffered_handle(
        &self,
        input: &MatrixBufferHandle,
        grad_out: &MatrixBufferHandle,
        params: &MatrixBufferView,
        grad_input: &MatrixBufferHandle,
        grad_params: &MatrixBufferView,
        seq_len: usize,
        input_dim: usize,
        state_dim: usize,
        h_all: &MatrixBufferHandle,
    ) {
        assert!(input.is_gpu(), "Input handle must be GPU");
        assert!(grad_out.is_gpu(), "grad_out handle must be GPU");
        assert!(grad_input.is_gpu(), "grad_input handle must be GPU");
        assert!(grad_params.is_gpu(), "grad_params view must point to GPU buffer");
        assert!(params.is_gpu(), "Params view must point to GPU buffer");

        let batch = input.rows();
        let d = input_dim;
        let n = state_dim;
        assert_eq!(input.cols(), seq_len * d, "Input cols mismatch");
        assert_eq!(grad_out.rows(), batch);
        assert_eq!(grad_out.cols(), seq_len * d, "grad_out cols mismatch");
        assert_eq!(grad_input.rows(), batch);
        assert_eq!(grad_input.cols(), seq_len * d, "grad_input cols mismatch");

        let param_len = n * n + n * d + d * n + 2;
        assert_eq!(params.len(), param_len, "Params length mismatch");
        assert_eq!(grad_params.len(), param_len, "Grad params length mismatch");
        assert_eq!(h_all.rows() * h_all.cols(), batch * seq_len * n, "h_all size mismatch");

        // Обнуляем градиенты по параметрам и входу
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
        self.fill_gpu_handle(grad_input, 0.0);

        // Создаём view для параметров и их градиентов
        let a_start = 0usize;
        let b_start = a_start + n * n;
        let c_start = b_start + n * d;
        let d_idx = c_start + d * n;
        let delta_idx = d_idx + 1;

        let a_view = MatrixBufferView::with_shape(
            params.parent_handle().clone(),
            params.offset_elements() + a_start,
            n * n,
            n,
            n,
        );
        let b_view = MatrixBufferView::with_shape(
            params.parent_handle().clone(),
            params.offset_elements() + b_start,
            n * d,
            n,
            d,
        );
        let c_view = MatrixBufferView::with_shape(
            params.parent_handle().clone(),
            params.offset_elements() + c_start,
            d * n,
            d,
            n,
        );
        let d_view = MatrixBufferView::new(
            params.parent_handle().clone(),
            params.offset_elements() + d_idx,
            1,
        );
        let delta_view = MatrixBufferView::new(
            params.parent_handle().clone(),
            params.offset_elements() + delta_idx,
            1,
        );

        let ga_view = MatrixBufferView::with_shape(
            grad_params.parent_handle().clone(),
            grad_params.offset_elements() + a_start,
            n * n,
            n,
            n,
        );
        let gb_view = MatrixBufferView::with_shape(
            grad_params.parent_handle().clone(),
            grad_params.offset_elements() + b_start,
            n * d,
            n,
            d,
        );
        let gc_view = MatrixBufferView::with_shape(
            grad_params.parent_handle().clone(),
            grad_params.offset_elements() + c_start,
            d * n,
            d,
            n,
        );
        let gd_view = MatrixBufferView::new(
            grad_params.parent_handle().clone(),
            grad_params.offset_elements() + d_idx,
            1,
        );
        let gdelta_view = MatrixBufferView::new(
            grad_params.parent_handle().clone(),
            grad_params.offset_elements() + delta_idx,
            1,
        );

        // Значения D и delta для push-констант и дискретизации
        let d_val = {
            let handle = self.download_gpu_handle_to_cpu_handle(d_view.parent_handle());
            let guard = handle.read();
            let slice = guard.as_slice().unwrap();
            slice[d_view.offset_elements()]
        };
        let delta_val = {
            let handle = self.download_gpu_handle_to_cpu_handle(delta_view.parent_handle());
            let guard = handle.read();
            let slice = guard.as_slice().unwrap();
            slice[delta_view.offset_elements()]
        };

        // Выделяем временные буферы A_bar, B_bar для дискретизации
        let (a_bar_buf, a_bar_raw) = self.acquire_temp_buffer(n * n);
        let (b_bar_buf, b_bar_raw) = self.acquire_temp_buffer(n * d);
        // Временные буферы для накопления градиентов A_bar, B_bar
        let (grad_a_bar_buf, grad_a_bar_raw) = self.acquire_temp_buffer(n * n);
        let (grad_b_bar_buf, grad_b_bar_raw) = self.acquire_temp_buffer(n * d);
        // Временные буферы delta_next
        let (delta_next_a_buf, delta_next_a_raw) = self.acquire_temp_buffer(batch * n);
        let (delta_next_b_buf, delta_next_b_raw) = self.acquire_temp_buffer(batch * n);

        // Обнуляем grad_A_bar, grad_B_bar и delta_next_a
        let zero_sub = self.upload_to_temp_buffer(&vec![0.0f32; grad_a_bar_buf.len() as usize]);
        self.copy_buffer_sync(zero_sub.0.clone(), grad_a_bar_buf.clone());
        self.copy_buffer_sync(zero_sub.0.clone(), grad_b_bar_buf.clone());
        self.copy_buffer_sync(zero_sub.0.clone(), delta_next_a_buf.clone());
        self.release_temp_buffer(zero_sub.0, zero_sub.1);

        // Запускаем дискретизацию
        let discretize_pipeline = &self.mamba_pipelines().discretize;
        let push_disc = [n as u32, d as u32, delta_val.to_bits(), 0u32];
        self.run_compute_shader(
            discretize_pipeline,
            &[
                (0, subbuffer_from_view(self, &a_view)),
                (1, subbuffer_from_view(self, &b_view)),
                (2, a_bar_buf.clone()),
                (3, b_bar_buf.clone()),
            ],
            &push_disc,
            n * n + n * d,
        );

        // Получаем нужные Subbuffer'ы
        let in_buf = self.get_gpu_subbuffer_from_handle(input);
        let go_buf = self.get_gpu_subbuffer_from_handle(grad_out);
        let h_buf = self.get_gpu_subbuffer_from_handle(h_all);
        let gi_buf = self.get_gpu_subbuffer_from_handle(grad_input);
        let c_buf = subbuffer_from_view(self, &c_view);
        let gc_buf = subbuffer_from_view(self, &gc_view);
        let gd_buf = subbuffer_from_view(self, &gd_view);

        // Цикл обратных шагов
        let bwd_pipeline = &self.mamba_pipelines().backward_step;
        // Используем два буфера для delta_next и меняем их местами
        let mut current_delta_in = delta_next_a_buf.clone();
        let mut current_delta_out = delta_next_b_buf.clone();

        for t in (0..seq_len).rev() {
            let push = [batch as u32, d as u32, n as u32, seq_len as u32, t as u32];
            self.run_compute_shader(
                bwd_pipeline,
                &[
                    (0, in_buf.clone()),
                    (1, go_buf.clone()),
                    (2, h_buf.clone()),
                    (3, a_bar_buf.clone()),
                    (4, b_bar_buf.clone()),
                    (5, c_buf.clone()),
                    (6, subbuffer_from_view(self, &d_view)),
                    (7, current_delta_in.clone()),
                    (8, grad_a_bar_buf.clone()),
                    (9, grad_b_bar_buf.clone()),
                    (10, gc_buf.clone()),
                    (11, gd_buf.clone()),
                    (12, gi_buf.clone()),
                    (13, current_delta_out.clone()),
                ],
                &push,
                batch,
            );
            // Меняем буферы местами для следующего шага
            std::mem::swap(&mut current_delta_in, &mut current_delta_out);
        }

        // Преобразуем градиенты A_bar, B_bar в градиенты A, B, delta
        let convert_pipeline = &self.mamba_pipelines().convert_grads;
        self.run_compute_shader(
            convert_pipeline,
            &[
                (0, subbuffer_from_view(self, &a_view)),
                (1, subbuffer_from_view(self, &b_view)),
                (2, grad_a_bar_buf.clone()),
                (3, grad_b_bar_buf.clone()),
                (4, subbuffer_from_view(self, &ga_view)),
                (5, subbuffer_from_view(self, &gb_view)),
                (6, subbuffer_from_view(self, &gdelta_view)),
            ],
            &push_disc,
            n * n + n * d,
        );

        // Освобождаем временные буферы
        self.release_temp_buffer(a_bar_buf, a_bar_raw);
        self.release_temp_buffer(b_bar_buf, b_bar_raw);
        self.release_temp_buffer(grad_a_bar_buf, grad_a_bar_raw);
        self.release_temp_buffer(grad_b_bar_buf, grad_b_bar_raw);
        self.release_temp_buffer(delta_next_a_buf, delta_next_a_raw);
        self.release_temp_buffer(delta_next_b_buf, delta_next_b_raw);
    }
}