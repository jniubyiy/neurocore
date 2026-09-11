// src/layers/adaptive_normalization/gpu/mod.rs

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
    /// Прямой проход AdaptiveNormalization на GPU.
    ///
    /// Этапы:
    ///   row_stats → col_stats → forward
    pub fn run_adaptive_norm_forward_buffered_handle(
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
        assert_eq!(params.len(), 7 * features, "Params length must be 7*features");

        // Временные буферы под статистики.
        let (row_mean_buf,   row_mean_raw)   = self.acquire_temp_buffer(batch);
        let (row_var_buf,    row_var_raw)    = self.acquire_temp_buffer(batch);
        let (row_rms_sq_buf, row_rms_sq_raw) = self.acquire_temp_buffer(batch);
        let (col_mean_buf,   col_mean_raw)   = self.acquire_temp_buffer(features);
        let (col_var_buf,    col_var_raw)    = self.acquire_temp_buffer(features);

        let in_buf     = self.get_gpu_subbuffer_from_handle(input);
        let params_buf = subbuffer_from_view(self, params);
        let out_buf    = self.get_gpu_subbuffer_from_handle(output);

        let push_stats = [batch as u32, features as u32];

        // 1. Статистики по строкам.
        self.run_compute_shader(
            &self.adaptive_normalization_pipelines().row_stats,
            &[
                (0, in_buf.clone()),
                (1, row_mean_buf.clone()),
                (2, row_var_buf.clone()),
                (3, row_rms_sq_buf.clone()),
            ],
            &push_stats,
            batch,
        );

        // 2. Статистики по столбцам.
        self.run_compute_shader(
            &self.adaptive_normalization_pipelines().col_stats,
            &[
                (0, in_buf.clone()),
                (1, col_mean_buf.clone()),
                (2, col_var_buf.clone()),
            ],
            &push_stats,
            features,
        );

        // 3. Основной forward.
        self.run_compute_shader(
            &self.adaptive_normalization_pipelines().forward,
            &[
                (0, in_buf.clone()),
                (1, out_buf),
                (2, params_buf),
                (3, row_mean_buf.clone()),
                (4, row_var_buf.clone()),
                (5, row_rms_sq_buf.clone()),
                (6, col_mean_buf.clone()),
                (7, col_var_buf.clone()),
            ],
            &push_stats,
            total,
        );

        self.release_temp_buffer(row_mean_buf, row_mean_raw);
        self.release_temp_buffer(row_var_buf, row_var_raw);
        self.release_temp_buffer(row_rms_sq_buf, row_rms_sq_raw);
        self.release_temp_buffer(col_mean_buf, col_mean_raw);
        self.release_temp_buffer(col_var_buf, col_var_raw);
    }

    /// Обратный проход AdaptiveNormalization на GPU.
    ///
    /// Этапы:
    ///   row_stats → col_stats → bwd_weights
    ///   → bwd_row_sums → bwd_col_sums
    ///   → bwd_input , bwd_params
    pub fn run_adaptive_norm_backward_buffered_handle(
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
        assert_eq!(params.len(), 7 * features, "Params length mismatch");
        assert_eq!(grad_params.len(), 7 * features, "grad_params length mismatch");

        // Временные буферы.
        let (row_mean_buf,   row_mean_raw)   = self.acquire_temp_buffer(batch);
        let (row_var_buf,    row_var_raw)    = self.acquire_temp_buffer(batch);
        let (row_rms_sq_buf, row_rms_sq_raw) = self.acquire_temp_buffer(batch);
        let (col_mean_buf,   col_mean_raw)   = self.acquire_temp_buffer(features);
        let (col_var_buf,    col_var_raw)    = self.acquire_temp_buffer(features);

        let (w_ln_buf,  w_ln_raw)  = self.acquire_temp_buffer(features);
        let (w_rms_buf, w_rms_raw) = self.acquire_temp_buffer(features);
        let (w_bn_buf,  w_bn_raw)  = self.acquire_temp_buffer(features);

        let (sum_dln_buf,    sum_dln_raw)    = self.acquire_temp_buffer(batch);
        let (sum_dln_x_buf,  sum_dln_x_raw)  = self.acquire_temp_buffer(batch);
        let (sum_drms_x_buf, sum_drms_x_raw) = self.acquire_temp_buffer(batch);

        let (sum_dbn_buf,   sum_dbn_raw)   = self.acquire_temp_buffer(features);
        let (sum_dbn_x_buf, sum_dbn_x_raw) = self.acquire_temp_buffer(features);

        let in_buf     = self.get_gpu_subbuffer_from_handle(input);
        let go_buf     = self.get_gpu_subbuffer_from_handle(grad_out);
        let gi_buf     = self.get_gpu_subbuffer_from_handle(grad_input);
        let params_buf = subbuffer_from_view(self, params);
        let grad_params_buf = subbuffer_from_view(self, grad_params);

        let push_stats = [batch as u32, features as u32];
        let push_feat  = [features as u32];

        // 1. Пересчёт статистик.
        self.run_compute_shader(
            &self.adaptive_normalization_pipelines().row_stats,
            &[
                (0, in_buf.clone()),
                (1, row_mean_buf.clone()),
                (2, row_var_buf.clone()),
                (3, row_rms_sq_buf.clone()),
            ],
            &push_stats,
            batch,
        );

        self.run_compute_shader(
            &self.adaptive_normalization_pipelines().col_stats,
            &[
                (0, in_buf.clone()),
                (1, col_mean_buf.clone()),
                (2, col_var_buf.clone()),
            ],
            &push_stats,
            features,
        );

        // 2. Веса softmax на признак.
        self.run_compute_shader(
            &self.adaptive_normalization_pipelines().bwd_weights,
            &[
                (0, params_buf.clone()),
                (1, w_ln_buf.clone()),
                (2, w_rms_buf.clone()),
                (3, w_bn_buf.clone()),
            ],
            &push_feat,
            features,
        );

        // 3. Суммы по строкам.
        self.run_compute_shader(
            &self.adaptive_normalization_pipelines().bwd_row_sums,
            &[
                (0, in_buf.clone()),
                (1, go_buf.clone()),
                (2, w_ln_buf.clone()),
                (3, w_rms_buf.clone()),
                (4, row_mean_buf.clone()),
                (5, sum_dln_buf.clone()),
                (6, sum_dln_x_buf.clone()),
                (7, sum_drms_x_buf.clone()),
            ],
            &push_stats,
            batch,
        );

        // 4. Суммы по столбцам.
        self.run_compute_shader(
            &self.adaptive_normalization_pipelines().bwd_col_sums,
            &[
                (0, in_buf.clone()),
                (1, go_buf.clone()),
                (2, w_bn_buf.clone()),
                (3, col_mean_buf.clone()),
                (4, sum_dbn_buf.clone()),
                (5, sum_dbn_x_buf.clone()),
            ],
            &push_stats,
            features,
        );

        // 5. Градиент по входу.
        self.run_compute_shader(
            &self.adaptive_normalization_pipelines().bwd_input,
            &[
                (0,  in_buf.clone()),
                (1,  go_buf.clone()),
                (2,  params_buf.clone()),
                (3,  row_mean_buf.clone()),
                (4,  row_var_buf.clone()),
                (5,  row_rms_sq_buf.clone()),
                (6,  col_mean_buf.clone()),
                (7,  col_var_buf.clone()),
                (8,  w_ln_buf.clone()),
                (9,  w_rms_buf.clone()),
                (10, w_bn_buf.clone()),
                (11, sum_dln_buf.clone()),
                (12, sum_dln_x_buf.clone()),
                (13, sum_drms_x_buf.clone()),
                (14, sum_dbn_buf.clone()),
                (15, sum_dbn_x_buf.clone()),
                (16, gi_buf),
            ],
            &push_stats,
            total,
        );

        // 6. Градиенты по параметрам.
        self.run_compute_shader(
            &self.adaptive_normalization_pipelines().bwd_params,
            &[
                (0,  in_buf.clone()),
                (1,  go_buf.clone()),
                (2,  params_buf),
                (3,  row_mean_buf.clone()),
                (4,  row_var_buf.clone()),
                (5,  row_rms_sq_buf.clone()),
                (6,  col_mean_buf.clone()),
                (7,  col_var_buf.clone()),
                (8,  w_ln_buf.clone()),
                (9,  w_rms_buf.clone()),
                (10, w_bn_buf.clone()),
                (11, grad_params_buf),
            ],
            &push_stats,
            features,
        );

        // Освобождение временных буферов.
        self.release_temp_buffer(row_mean_buf, row_mean_raw);
        self.release_temp_buffer(row_var_buf, row_var_raw);
        self.release_temp_buffer(row_rms_sq_buf, row_rms_sq_raw);
        self.release_temp_buffer(col_mean_buf, col_mean_raw);
        self.release_temp_buffer(col_var_buf, col_var_raw);

        self.release_temp_buffer(w_ln_buf, w_ln_raw);
        self.release_temp_buffer(w_rms_buf, w_rms_raw);
        self.release_temp_buffer(w_bn_buf, w_bn_raw);

        self.release_temp_buffer(sum_dln_buf, sum_dln_raw);
        self.release_temp_buffer(sum_dln_x_buf, sum_dln_x_raw);
        self.release_temp_buffer(sum_drms_x_buf, sum_drms_x_raw);

        self.release_temp_buffer(sum_dbn_buf, sum_dbn_raw);
        self.release_temp_buffer(sum_dbn_x_buf, sum_dbn_x_raw);
    }
}