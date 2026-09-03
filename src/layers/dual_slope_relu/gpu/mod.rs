// src/layers/dual_slope_relu/gpu/mod.rs

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
    /// Прямой проход DualSlopeReLU на GPU.
    pub fn run_dual_slope_relu_forward_buffered_handle(
        &self,
        input: &MatrixBufferHandle,
        alpha: &MatrixBufferView,
        beta: &MatrixBufferView,
        output: &MatrixBufferHandle,
    ) {
        assert!(input.is_gpu(), "Input handle must be GPU");
        assert!(output.is_gpu(), "Output handle must be GPU");
        assert!(alpha.is_gpu(), "Alpha view must point to GPU buffer");
        assert!(beta.is_gpu(), "Beta view must point to GPU buffer");

        let batch = input.rows();
        let features = input.cols();
        let total = batch * features;
        assert_eq!(output.rows(), batch);
        assert_eq!(output.cols(), features);
        assert_eq!(alpha.len(), features, "Alpha length must equal features");
        assert_eq!(beta.len(), features, "Beta length must equal features");

        let in_buf = self.get_gpu_subbuffer_from_handle(input);
        let alpha_buf = subbuffer_from_view(self, alpha);
        let beta_buf = subbuffer_from_view(self, beta);
        let out_buf = self.get_gpu_subbuffer_from_handle(output);

        let pipeline = &self.dual_slope_relu_pipelines().forward;
        let push = [total as u32, features as u32];
        self.run_compute_shader(
            pipeline,
            &[(0, in_buf), (1, alpha_buf), (2, beta_buf), (3, out_buf)],
            &push,
            total,
        );
    }

    /// Обратный проход DualSlopeReLU на GPU.
    pub fn run_dual_slope_relu_backward_buffered_handle(
        &self,
        input: &MatrixBufferHandle,
        grad_out: &MatrixBufferHandle,
        alpha: &MatrixBufferView,
        beta: &MatrixBufferView,
        grad_input: &MatrixBufferHandle,
        grad_alpha: &MatrixBufferView,
        grad_beta: &MatrixBufferView,
    ) {
        assert!(input.is_gpu(), "Input handle must be GPU");
        assert!(grad_out.is_gpu(), "grad_out handle must be GPU");
        assert!(grad_input.is_gpu(), "grad_input handle must be GPU");
        assert!(grad_alpha.is_gpu(), "grad_alpha view must point to GPU buffer");
        assert!(grad_beta.is_gpu(), "grad_beta view must point to GPU buffer");
        assert!(alpha.is_gpu(), "Alpha view must point to GPU buffer");
        assert!(beta.is_gpu(), "Beta view must point to GPU buffer");

        let batch = input.rows();
        let features = input.cols();
        let total = batch * features;
        assert_eq!(grad_out.rows(), batch);
        assert_eq!(grad_out.cols(), features);
        assert_eq!(grad_input.rows(), batch);
        assert_eq!(grad_input.cols(), features);
        assert_eq!(grad_alpha.len(), features, "grad_alpha length must equal features");
        assert_eq!(grad_beta.len(), features, "grad_beta length must equal features");

        let in_buf = self.get_gpu_subbuffer_from_handle(input);
        let go_buf = self.get_gpu_subbuffer_from_handle(grad_out);
        let alpha_buf = subbuffer_from_view(self, alpha);
        let beta_buf = subbuffer_from_view(self, beta);
        let gi_buf = self.get_gpu_subbuffer_from_handle(grad_input);
        let grad_alpha_buf = subbuffer_from_view(self, grad_alpha);
        let grad_beta_buf = subbuffer_from_view(self, grad_beta);

        let pipeline = &self.dual_slope_relu_pipelines().backward;
        let push = [total as u32, features as u32];
        self.run_compute_shader(
            pipeline,
            &[
                (0, in_buf),
                (1, go_buf),
                (2, alpha_buf),
                (3, beta_buf),
                (4, gi_buf),
                (5, grad_alpha_buf),
                (6, grad_beta_buf),
            ],
            &push,
            total,
        );
    }
}