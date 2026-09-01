// src/layers/linear/gpu/mod.rs

pub mod pipeline;

use crate::compute_manager::gpu::compute::GpuCompute;
use crate::compute_manager::matrix_buffer::view::MatrixBufferView;
use crate::compute_manager::matrix_buffer::MatrixBufferHandle;
use vulkano::buffer::Subbuffer;

/// Вспомогательная функция: получает `Subbuffer<[f32]>` из `MatrixBufferView`,
/// используя смещение и длину. Родительский буфер должен быть GPU.
fn subbuffer_from_view(gpu: &GpuCompute, view: &MatrixBufferView) -> Subbuffer<[f32]> {
    let parent_sub = gpu.get_gpu_subbuffer_from_handle(view.parent_handle());
    let start = view.offset_elements() as u64;
    let end = (view.offset_elements() + view.len()) as u64;
    parent_sub.slice(start..end)
}

impl GpuCompute {
    /// Прямой проход линейного слоя на GPU.
    ///
    /// Веса и смещения передаются как `MatrixBufferView`, которые ссылаются
    /// на части одного большого GPU-буфера параметров сегмента.
    /// Вход и выход — GPU-дескрипторы.
    pub fn run_linear_forward_buffered_handle(
        &self,
        input: &MatrixBufferHandle,
        weight: &MatrixBufferView,
        bias: &MatrixBufferView,
        output: &MatrixBufferHandle,
    ) {
        assert!(input.is_gpu(), "Input handle must be GPU");
        assert!(output.is_gpu(), "Output handle must be GPU");
        assert!(weight.is_gpu(), "Weight view must point to GPU buffer");
        assert!(bias.is_gpu(), "Bias view must point to GPU buffer");

        let batch = input.rows();
        let in_features = input.cols();
        let out_features = weight.rows();
        assert_eq!(weight.cols(), in_features, "Weight shape mismatch");
        assert_eq!(output.rows(), batch, "Output rows mismatch");
        assert_eq!(output.cols(), out_features, "Output cols mismatch");
        assert_eq!(bias.len(), out_features, "Bias length mismatch");

        let in_buf = self.get_gpu_subbuffer_from_handle(input);
        let w_buf = subbuffer_from_view(self, weight);
        let b_buf = subbuffer_from_view(self, bias);
        let out_buf = self.get_gpu_subbuffer_from_handle(output);

        // Используем пайплайн линейного слоя (веса и смещения уже на GPU)
        let pipeline = &self.linear_pipelines().forward;
        let push = [batch as u32, in_features as u32, out_features as u32];

        self.run_compute_shader_with_dispatch(
            pipeline,
            &[(0, in_buf), (1, w_buf), (2, b_buf), (3, out_buf)],
            &push,
            [((batch * out_features + 255) / 256) as u32, 1, 1],
        );
    }

    /// Обратный проход линейного слоя на GPU.
    ///
    /// Градиенты весов и смещений записываются непосредственно в
    /// предоставленные `MatrixBufferView` (части общего буфера градиентов).
    /// Входной градиент (`grad_output`) и выходной (`grad_input`) — GPU-дескрипторы.
    pub fn run_linear_backward_buffered_handle(
        &self,
        input: &MatrixBufferHandle,
        weight: &MatrixBufferView,
        grad_output: &MatrixBufferHandle,
        grad_input: &MatrixBufferHandle,
        grad_weight: &MatrixBufferView,
        grad_bias: &MatrixBufferView,
    ) {
        assert!(input.is_gpu(), "Input handle must be GPU");
        assert!(weight.is_gpu(), "Weight view must point to GPU buffer");
        assert!(grad_output.is_gpu(), "grad_output handle must be GPU");
        assert!(grad_input.is_gpu(), "grad_input handle must be GPU");
        assert!(grad_weight.is_gpu(), "grad_weight view must point to GPU buffer");
        assert!(grad_bias.is_gpu(), "grad_bias view must point to GPU buffer");

        let batch = input.rows();
        let in_features = input.cols();
        let out_features = weight.rows();
        assert_eq!(weight.cols(), in_features, "Weight shape mismatch");
        assert_eq!(grad_input.rows(), batch, "grad_input rows mismatch");
        assert_eq!(grad_input.cols(), in_features, "grad_input cols mismatch");
        assert_eq!(grad_weight.rows(), out_features, "grad_weight rows mismatch");
        assert_eq!(grad_weight.cols(), in_features, "grad_weight cols mismatch");
        assert_eq!(grad_output.rows(), batch, "grad_output rows mismatch");
        assert_eq!(grad_output.cols(), out_features, "grad_output cols mismatch");
        assert_eq!(grad_bias.len(), out_features, "grad_bias length mismatch");

        let x_buf = self.get_gpu_subbuffer_from_handle(input);
        let w_buf = subbuffer_from_view(self, weight);
        let dout_buf = self.get_gpu_subbuffer_from_handle(grad_output);
        let dx_buf = self.get_gpu_subbuffer_from_handle(grad_input);
        let dw_buf = subbuffer_from_view(self, grad_weight);
        let db_buf = subbuffer_from_view(self, grad_bias);

        // Используем пайплайн обратного прохода линейного слоя.
        let pipeline = &self.linear_pipelines().backward;
        let push = [batch as u32, in_features as u32, out_features as u32];

        self.run_compute_shader_with_dispatch(
            pipeline,
            &[
                (0, x_buf),
                (1, w_buf),
                (2, dout_buf),
                (3, dx_buf),
                (4, dw_buf),
                (5, db_buf),
            ],
            &push,
            [((batch + 255) / 256) as u32, 1, 1],
        );
    }
}