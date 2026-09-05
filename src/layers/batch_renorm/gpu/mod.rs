// src/layers/batch_renorm/gpu/mod.rs

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
    /// Прямой проход BatchRenorm1d на GPU.
    ///
    /// Параметры (4*features элементов: gamma, beta, r, d) передаются как `MatrixBufferView`,
    /// ссылающийся на часть общего GPU-буфера параметров сегмента.
    /// Вход и выход — GPU-дескрипторы.
    /// Внутри вычисляются статистики по столбцам (mean, var) через отдельный
    /// редукционный шейдер, после чего вызывается основной forward-шейдер.
    pub fn run_batch_renorm_forward_buffered_handle(
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
        assert_eq!(params.len(), 4 * features, "Params length must be 4*features");

        // Выделяем временные буферы для статистик
        let (col_mean_buf, col_mean_raw) = self.acquire_temp_buffer(features);
        let (col_var_buf, col_var_raw) = self.acquire_temp_buffer(features);

        // Получаем Subbuffer для входного тензора и параметров
        let in_buf = self.get_gpu_subbuffer_from_handle(input);
        let params_buf = subbuffer_from_view(self, params);
        let out_buf = self.get_gpu_subbuffer_from_handle(output);

        // Запускаем редукционный шейдер для статистик
        let col_stats_pipeline = &self.batch_renorm_pipelines().col_stats;
        let push_stats = [batch as u32, features as u32];
        self.run_compute_shader(
            col_stats_pipeline,
            &[
                (0, in_buf.clone()),
                (1, col_mean_buf.clone()),
                (2, col_var_buf.clone()),
            ],
            &push_stats,
            features,
        );

        // Запускаем основной forward-шейдер
        let forward_pipeline = &self.batch_renorm_pipelines().forward;
        let push_fwd = [batch as u32, features as u32];
        self.run_compute_shader(
            forward_pipeline,
            &[
                (0, in_buf.clone()),
                (1, params_buf),
                (2, col_mean_buf.clone()),
                (3, col_var_buf.clone()),
                (4, out_buf),
            ],
            &push_fwd,
            total,
        );

        // Освобождаем временные буферы
        self.release_temp_buffer(col_mean_buf, col_mean_raw);
        self.release_temp_buffer(col_var_buf, col_var_raw);
    }

    /// Обратный проход BatchRenorm1d на GPU.
    ///
    /// Градиенты по параметрам записываются в `grad_params` (часть общего GPU-буфера
    /// градиентов). Вход/выходные градиенты — GPU-дескрипторы.
    /// Перед вызовом область `grad_params` обнуляется, так как шейдер использует
    /// атомарное накопление. Статистики пересчитываются заново.
    pub fn run_batch_renorm_backward_buffered_handle(
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
        assert_eq!(params.len(), 4 * features, "Params length mismatch");
        assert_eq!(grad_params.len(), 4 * features, "grad_params length mismatch");

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

        // Выделяем временные буферы для статистик
        let (col_mean_buf, col_mean_raw) = self.acquire_temp_buffer(features);
        let (col_var_buf, col_var_raw) = self.acquire_temp_buffer(features);

        // Получаем нужные Subbuffer'ы
        let in_buf = self.get_gpu_subbuffer_from_handle(input);
        let go_buf = self.get_gpu_subbuffer_from_handle(grad_out);
        let gi_buf = self.get_gpu_subbuffer_from_handle(grad_input);
        let params_buf = subbuffer_from_view(self, params);
        let grad_params_buf = subbuffer_from_view(self, grad_params);

        // Пересчитываем статистики
        let col_stats_pipeline = &self.batch_renorm_pipelines().col_stats;
        let push_stats = [batch as u32, features as u32];
        self.run_compute_shader(
            col_stats_pipeline,
            &[
                (0, in_buf.clone()),
                (1, col_mean_buf.clone()),
                (2, col_var_buf.clone()),
            ],
            &push_stats,
            features,
        );

        // Запускаем основной backward-шейдер
        let backward_pipeline = &self.batch_renorm_pipelines().backward;
        let push_bwd = [batch as u32, features as u32];
        self.run_compute_shader(
            backward_pipeline,
            &[
                (0, in_buf.clone()),
                (1, go_buf),
                (2, params_buf),
                (3, col_mean_buf.clone()),
                (4, col_var_buf.clone()),
                (5, gi_buf),
                (6, grad_params_buf),
            ],
            &push_bwd,
            total,
        );

        // Освобождаем временные буферы
        self.release_temp_buffer(col_mean_buf, col_mean_raw);
        self.release_temp_buffer(col_var_buf, col_var_raw);
    }
}