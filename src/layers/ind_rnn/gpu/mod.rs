// src/layers/ind_rnn/gpu/mod.rs

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
    /// Прямой проход IndRNN на GPU.
    ///
    /// Параметры (W: input_dim×input_dim, u: input_dim, b: input_dim) передаются как
    /// `MatrixBufferView` на полный блок. Вход/выход — GPU-дескрипторы (column-major).
    /// Скрытые состояния h_all хранятся в row-major: (r * seq_len + t) * input_dim + i.
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
        assert!(h_all.is_gpu(), "h_all handle must be GPU");

        let batch = input.rows();
        assert_eq!(input.cols(), seq_len * input_dim, "Input cols mismatch");
        assert_eq!(output.rows(), batch);
        assert_eq!(output.cols(), seq_len * input_dim, "Output cols mismatch");
        assert_eq!(
            h_all.rows() * h_all.cols(),
            batch * seq_len * input_dim,
            "h_all size mismatch"
        );

        let d = input_dim;
        let param_len = d * d + 2 * d;
        assert_eq!(params.len(), param_len, "Params length mismatch");

        // Создаём view для W, u, b
        let w_view = MatrixBufferView::with_shape(
            params.parent_handle().clone(),
            params.offset_elements(),
            d * d,
            d,
            d,
        );
        let u_view = MatrixBufferView::new(
            params.parent_handle().clone(),
            params.offset_elements() + d * d,
            d,
        );
        let b_view = MatrixBufferView::new(
            params.parent_handle().clone(),
            params.offset_elements() + d * d + d,
            d,
        );

        let in_buf = self.get_gpu_subbuffer_from_handle(input);
        let out_buf = self.get_gpu_subbuffer_from_handle(output);
        let h_buf = self.get_gpu_subbuffer_from_handle(h_all);

        let w_buf = subbuffer_from_view(self, &w_view);
        let u_buf = subbuffer_from_view(self, &u_view);
        let b_buf = subbuffer_from_view(self, &b_view);

        let fwd_pipeline = &self.ind_rnn_pipelines().forward_step;

        for t in 0..seq_len {
            let push = [
                batch as u32,
                d as u32,
                seq_len as u32,
                t as u32,
            ];
            self.run_compute_shader(
                fwd_pipeline,
                &[
                    (0, in_buf.clone()),
                    (1, w_buf.clone()),
                    (2, u_buf.clone()),
                    (3, b_buf.clone()),
                    (4, h_buf.clone()),
                    (5, out_buf.clone()),
                ],
                &push,
                batch * d,
            );
        }
    }

    /// Обратный проход IndRNN на GPU.
    ///
    /// Градиенты по параметрам записываются в `grad_params` (view на W,u,b) атомарно.
    /// Вход/выходные градиенты — GPU-дескрипторы (column-major).
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
        assert!(h_all.is_gpu(), "h_all handle must be GPU");

        let batch = input.rows();
        let d = input_dim;
        let param_len = d * d + 2 * d;
        assert_eq!(grad_out.rows(), batch);
        assert_eq!(grad_out.cols(), seq_len * d);
        assert_eq!(grad_input.rows(), batch);
        assert_eq!(grad_input.cols(), seq_len * d);
        assert_eq!(params.len(), param_len, "Params length mismatch");
        assert_eq!(grad_params.len(), param_len, "Grad params length mismatch");
        assert_eq!(
            h_all.rows() * h_all.cols(),
            batch * seq_len * d,
            "h_all size mismatch"
        );

        // Обнуляем grad_params
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
        // Обнуляем grad_input
        self.fill_gpu_handle(grad_input, 0.0);

        // Создаём view для W,u,b и их градиентов
        let w_view = MatrixBufferView::with_shape(
            params.parent_handle().clone(),
            params.offset_elements(),
            d * d,
            d,
            d,
        );
        let u_view = MatrixBufferView::new(
            params.parent_handle().clone(),
            params.offset_elements() + d * d,
            d,
        );
        let b_view = MatrixBufferView::new(
            params.parent_handle().clone(),
            params.offset_elements() + d * d + d,
            d,
        );

        let gw_view = MatrixBufferView::with_shape(
            grad_params.parent_handle().clone(),
            grad_params.offset_elements(),
            d * d,
            d,
            d,
        );
        let gu_view = MatrixBufferView::new(
            grad_params.parent_handle().clone(),
            grad_params.offset_elements() + d * d,
            d,
        );
        let gb_view = MatrixBufferView::new(
            grad_params.parent_handle().clone(),
            grad_params.offset_elements() + d * d + d,
            d,
        );

        let in_buf = self.get_gpu_subbuffer_from_handle(input);
        let go_buf = self.get_gpu_subbuffer_from_handle(grad_out);
        let gi_buf = self.get_gpu_subbuffer_from_handle(grad_input);
        let h_buf = self.get_gpu_subbuffer_from_handle(h_all);

        let w_buf = subbuffer_from_view(self, &w_view);
        let u_buf = subbuffer_from_view(self, &u_view);
        let b_buf = subbuffer_from_view(self, &b_view);
        let gw_buf = subbuffer_from_view(self, &gw_view);
        let gu_buf = subbuffer_from_view(self, &gu_view);
        let gb_buf = subbuffer_from_view(self, &gb_view);

        // Временные буферы для delta_next (ping-pong)
        let (delta_next_a_buf, delta_next_a_raw) = self.acquire_temp_buffer(batch * d);
        let (delta_next_b_buf, delta_next_b_raw) = self.acquire_temp_buffer(batch * d);

        // Инициализируем delta_next_a нулями
        let zero_delta = self.upload_vec_to_gpu_handle(&vec![0.0f32; batch * d], batch * d, 1);
        self.copy_buffer_sync(
            self.get_gpu_subbuffer_from_handle(&zero_delta),
            delta_next_a_buf.clone(),
        );

        let bwd_pipeline = &self.ind_rnn_pipelines().backward_step;

        let mut current_delta_in = delta_next_a_buf.clone();
        let mut current_delta_out = delta_next_b_buf.clone();

        for t in (0..seq_len).rev() {
            let push = [
                batch as u32,
                d as u32,
                seq_len as u32,
                t as u32,
            ];
            self.run_compute_shader(
                bwd_pipeline,
                &[
                    (0, in_buf.clone()),
                    (1, w_buf.clone()),
                    (2, u_buf.clone()),
                    (3, b_buf.clone()),
                    (4, go_buf.clone()),
                    (5, h_buf.clone()),
                    (6, current_delta_in.clone()),
                    (7, gw_buf.clone()),
                    (8, gu_buf.clone()),
                    (9, gb_buf.clone()),
                    (10, gi_buf.clone()),
                    (11, current_delta_out.clone()),
                ],
                &push,
                batch * d,
            );
            std::mem::swap(&mut current_delta_in, &mut current_delta_out);
        }

        // Освобождаем временные буферы
        self.release_temp_buffer(delta_next_a_buf, delta_next_a_raw);
        self.release_temp_buffer(delta_next_b_buf, delta_next_b_raw);
    }
}