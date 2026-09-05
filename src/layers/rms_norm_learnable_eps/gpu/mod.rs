// src/layers/rms_norm_learnable_eps/gpu/mod.rs

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
    /// Прямой проход RMSNormWithLearnableEpsilon на GPU.
    ///
    /// Параметры (2*features элементов: gamma, eps) передаются как `MatrixBufferView`,
    /// ссылающийся на часть общего GPU-буфера параметров сегмента.
    /// Вход и выход — GPU-дескрипторы.
    /// Внутри вычисляется статистика среднего квадрата по каждой строке через
    /// отдельный редукционный шейдер, затем вызывается основной forward-шейдер.
    pub fn run_rms_norm_learnable_eps_forward_buffered_handle(
        &self,
        input: &MatrixBufferHandle,
        params: &MatrixBufferView,
        output: &MatrixBufferHandle,
    ) {
        assert!(input.is_gpu(), "Input handle must be GPU");
        assert!(output.is_gpu(), "Output handle must be GPU");
        assert!(params.is_gpu(), "Params view must point to GPU buffer");

        let batch = input.rows();
        let features = input.cols();
        let total = batch * features;
        assert_eq!(output.rows(), batch);
        assert_eq!(output.cols(), features);
        assert_eq!(params.len(), 2 * features, "Params length must be 2*features");

        // Выделяем временный буфер для статистики
        let (row_rms_sq_buf, row_rms_sq_raw) = self.acquire_temp_buffer(batch);

        // Получаем Subbuffer для входного тензора и параметров
        let in_buf = self.get_gpu_subbuffer_from_handle(input);
        let params_buf = subbuffer_from_view(self, params);
        let out_buf = self.get_gpu_subbuffer_from_handle(output);

        // Запускаем редукционный шейдер для row_rms_sq
        let row_stats_pipeline = &self.rms_norm_learnable_eps_pipelines().row_stats;
        let push_stats = [batch as u32, features as u32];
        self.run_compute_shader(
            row_stats_pipeline,
            &[
                (0, in_buf.clone()),
                (1, row_rms_sq_buf.clone()),
            ],
            &push_stats,
            batch,
        );

        // Запускаем основной forward-шейдер
        let forward_pipeline = &self.rms_norm_learnable_eps_pipelines().forward;
        let push_fwd = [batch as u32, features as u32];
        self.run_compute_shader(
            forward_pipeline,
            &[
                (0, in_buf.clone()),
                (1, params_buf),
                (2, row_rms_sq_buf.clone()),
                (3, out_buf),
            ],
            &push_fwd,
            total,
        );

        // Освобождаем временный буфер
        self.release_temp_buffer(row_rms_sq_buf, row_rms_sq_raw);
    }

    /// Обратный проход RMSNormWithLearnableEpsilon на GPU.
    ///
    /// Градиенты по параметрам записываются в `grad_params` (часть общего GPU-буфера
    /// градиентов). Вход/выходные градиенты — GPU-дескрипторы.
    /// Перед вызовом область `grad_params` обнуляется, так как шейдер использует
    /// атомарное накопление. Статистика среднего квадрата пересчитывается.
    pub fn run_rms_norm_learnable_eps_backward_buffered_handle(
        &self,
        input: &MatrixBufferHandle,
        grad_out: &MatrixBufferHandle,
        params: &MatrixBufferView,
        grad_input: &MatrixBufferHandle,
        grad_params: &MatrixBufferView,
    ) {
        assert!(input.is_gpu(), "Input handle must be GPU");
        assert!(grad_out.is_gpu(), "grad_out handle must be GPU");
        assert!(grad_input.is_gpu(), "grad_input handle must be GPU");
        assert!(grad_params.is_gpu(), "grad_params view must point to GPU buffer");
        assert!(params.is_gpu(), "Params view must point to GPU buffer");

        let batch = input.rows();
        let features = input.cols();
        let total = batch * features;
        assert_eq!(grad_out.rows(), batch);
        assert_eq!(grad_out.cols(), features);
        assert_eq!(grad_input.rows(), batch);
        assert_eq!(grad_input.cols(), features);
        assert_eq!(params.len(), 2 * features, "Params length mismatch");
        assert_eq!(grad_params.len(), 2 * features, "grad_params length mismatch");

        // Обнуляем область grad_params перед накоплением
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
        // zero_handle выйдет из области видимости и будет освобождён

        // Выделяем временный буфер для статистики
        let (row_rms_sq_buf, row_rms_sq_raw) = self.acquire_temp_buffer(batch);

        // Получаем нужные Subbuffer'ы
        let in_buf = self.get_gpu_subbuffer_from_handle(input);
        let go_buf = self.get_gpu_subbuffer_from_handle(grad_out);
        let gi_buf = self.get_gpu_subbuffer_from_handle(grad_input);
        let params_buf = subbuffer_from_view(self, params);
        let grad_params_buf = subbuffer_from_view(self, grad_params);

        // Пересчитываем статистику
        let row_stats_pipeline = &self.rms_norm_learnable_eps_pipelines().row_stats;
        let push_stats = [batch as u32, features as u32];
        self.run_compute_shader(
            row_stats_pipeline,
            &[
                (0, in_buf.clone()),
                (1, row_rms_sq_buf.clone()),
            ],
            &push_stats,
            batch,
        );

        // Запускаем основной backward-шейдер
        let backward_pipeline = &self.rms_norm_learnable_eps_pipelines().backward;
        let push_bwd = [batch as u32, features as u32];
        self.run_compute_shader(
            backward_pipeline,
            &[
                (0, in_buf.clone()),
                (1, go_buf),
                (2, params_buf),
                (3, gi_buf),
                (4, grad_params_buf),
                (5, row_rms_sq_buf.clone()),
            ],
            &push_bwd,
            total,
        );

        // Освобождаем временный буфер
        self.release_temp_buffer(row_rms_sq_buf, row_rms_sq_raw);
    }
}