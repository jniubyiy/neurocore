// src/layers/feature_fusion/gpu/mod.rs

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
    /// Прямой проход FeatureFusion на GPU.
    ///
    /// Разбит на два этапа:
    ///   1. softmax логитов → временный буфер `weights` (row-major, out × in);
    ///   2. y = W_softmax · x + b.
    ///
    /// Временный буфер выделяется через `acquire_temp_buffer` и освобождается
    /// в конце. Локальных массивов в шейдерах нет, ограничений на размерности нет.
    pub fn run_feature_fusion_forward_buffered_handle(
        &self,
        input: &MatrixBufferHandle,
        params: &MatrixBufferView,
        output: &MatrixBufferHandle,
    ) {
        assert!(input.is_gpu(), "Input handle must be GPU");
        assert!(output.is_gpu(), "Output handle must be GPU");
        assert!(params.is_gpu(), "Params view must point to GPU buffer");

        let batch = input.rows();
        let in_features = input.cols();
        let out_features = output.cols();
        assert_eq!(output.rows(), batch, "Output rows mismatch");
        assert_eq!(
            params.len(),
            out_features * (in_features + 1),
            "Params length mismatch"
        );

        // 1. Временный буфер под softmax-веса: row-major (out_features × in_features).
        let weights_elems = out_features * in_features;
        let (weights_buf, weights_raw) = self.acquire_temp_buffer(weights_elems);

        let in_buf = self.get_gpu_subbuffer_from_handle(input);
        let params_buf = subbuffer_from_view(self, params);
        let out_buf = self.get_gpu_subbuffer_from_handle(output);

        // 2. Softmax логитов.
        let softmax_pipeline = &self.feature_fusion_pipelines().softmax;
        let push_softmax = [out_features as u32, in_features as u32];
        self.run_compute_shader(
            softmax_pipeline,
            &[(0, params_buf.clone()), (1, weights_buf.clone())],
            &push_softmax,
            out_features,
        );

        // 3. Основной forward: matmul с softmax-весами + bias.
        let output_pipeline = &self.feature_fusion_pipelines().output;
        let push_output = [batch as u32, in_features as u32, out_features as u32];
        self.run_compute_shader(
            output_pipeline,
            &[
                (0, in_buf),
                (1, params_buf),
                (2, weights_buf.clone()),
                (3, out_buf),
            ],
            &push_output,
            batch * out_features,
        );

        // 4. Освобождаем временный буфер.
        self.release_temp_buffer(weights_buf, weights_raw);
    }

    /// Обратный проход FeatureFusion на GPU.
    ///
    /// Разбит на три этапа:
    ///   1. softmax логитов (тот же шейдер, что и в forward);
    ///   2. gi = go · W_softmax;
    ///   3. grad_logits и grad_bias.
    ///
    /// Временный буфер под softmax-веса освобождается в конце.
    pub fn run_feature_fusion_backward_buffered_handle(
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
        assert!(params.is_gpu(), "Params view must point to GPU buffer");
        assert!(grad_params.is_gpu(), "grad_params view must point to GPU buffer");

        let batch = input.rows();
        let in_features = input.cols();
        let out_features = grad_out.cols();
        assert_eq!(grad_out.rows(), batch, "grad_out rows mismatch");
        assert_eq!(grad_input.rows(), batch, "grad_input rows mismatch");
        assert_eq!(grad_input.cols(), in_features, "grad_input cols mismatch");
        assert_eq!(
            params.len(),
            out_features * (in_features + 1),
            "Params length mismatch"
        );
        assert_eq!(
            grad_params.len(),
            out_features * (in_features + 1),
            "grad_params length mismatch"
        );

        // 1. Временный буфер под softmax-веса.
        let weights_elems = out_features * in_features;
        let (weights_buf, weights_raw) = self.acquire_temp_buffer(weights_elems);

        let in_buf = self.get_gpu_subbuffer_from_handle(input);
        let go_buf = self.get_gpu_subbuffer_from_handle(grad_out);
        let gi_buf = self.get_gpu_subbuffer_from_handle(grad_input);
        let params_buf = subbuffer_from_view(self, params);
        let grad_params_buf = subbuffer_from_view(self, grad_params);

        // 2. Softmax логитов (тот же шейдер).
        let softmax_pipeline = &self.feature_fusion_pipelines().softmax;
        let push_softmax = [out_features as u32, in_features as u32];
        self.run_compute_shader(
            softmax_pipeline,
            &[(0, params_buf), (1, weights_buf.clone())],
            &push_softmax,
            out_features,
        );

        // 3. Градиент по входу.
        let grad_in_pipeline = &self.feature_fusion_pipelines().grad_input;
        let push_dims = [batch as u32, in_features as u32, out_features as u32];
        self.run_compute_shader(
            grad_in_pipeline,
            &[(0, go_buf.clone()), (1, weights_buf.clone()), (2, gi_buf)],
            &push_dims,
            batch * in_features,
        );

        // 4. Градиенты по параметрам.
        //
        // Разбиваем grad_params на два диапазона:
        //   [0, out*in)          → grad_logits (row-major)
        //   [out*in, out*in+out) → grad_bias
        //
        // Так как внутри grad_params лежат [logits; bias], эти два view
        // покрывают его без пересечения.
        let grad_logits_view = MatrixBufferView::new(
            grad_params.parent_handle().clone(),
            grad_params.offset_elements(),
            weights_elems,
        );
        let grad_bias_view = MatrixBufferView::new(
            grad_params.parent_handle().clone(),
            grad_params.offset_elements() + weights_elems,
            out_features,
        );

        let grad_logits_buf = subbuffer_from_view(self, &grad_logits_view);
        let grad_bias_buf = subbuffer_from_view(self, &grad_bias_view);

        let grad_params_pipeline = &self.feature_fusion_pipelines().grad_params;
        let total_params = weights_elems + out_features;
        self.run_compute_shader(
            grad_params_pipeline,
            &[
                (0, in_buf),
                (1, go_buf),
                (2, weights_buf.clone()),
                (3, grad_logits_buf),
                (4, grad_bias_buf),
            ],
            &push_dims,
            total_params,
        );

        // Освобождаем временный буфер.
        self.release_temp_buffer(weights_buf, weights_raw);

        // grad_params_buf не используется напрямую — он покрыт через два view,
        // но чтобы избежать предупреждения о неиспользуемой переменной,
        // оставим явную ссылку (компилятор её уберёт).
        let _ = grad_params_buf;
    }
}