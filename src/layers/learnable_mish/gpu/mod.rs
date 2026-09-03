// src/layers/learnable_mish/gpu/mod.rs

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
    /// Прямой проход LearnableMish на GPU.
    pub fn run_learnable_mish_forward_buffered_handle(
        &self,
        input: &MatrixBufferHandle,
        lambda: &MatrixBufferView,
        output: &MatrixBufferHandle,
    ) {
        assert!(input.is_gpu(), "Input handle must be GPU");
        assert!(output.is_gpu(), "Output handle must be GPU");
        assert!(lambda.is_gpu(), "Lambda view must point to GPU buffer");
        assert_eq!(lambda.len(), 1, "Lambda must be a single scalar");

        let batch = input.rows();
        let features = input.cols();
        let total = batch * features;
        assert_eq!(output.rows(), batch);
        assert_eq!(output.cols(), features);

        let in_buf = self.get_gpu_subbuffer_from_handle(input);
        let lambda_buf = subbuffer_from_view(self, lambda);
        let out_buf = self.get_gpu_subbuffer_from_handle(output);

        let pipeline = &self.learnable_mish_pipelines().forward;
        let push = [total as u32];
        self.run_compute_shader(
            pipeline,
            &[(0, in_buf), (1, lambda_buf), (2, out_buf)],
            &push,
            total,
        );
    }

    /// Обратный проход LearnableMish на GPU.
    pub fn run_learnable_mish_backward_buffered_handle(
        &self,
        input: &MatrixBufferHandle,
        grad_out: &MatrixBufferHandle,
        lambda: &MatrixBufferView,
        grad_input: &MatrixBufferHandle,
        grad_lambda: &MatrixBufferView,
    ) {
        assert!(input.is_gpu(), "Input handle must be GPU");
        assert!(grad_out.is_gpu(), "grad_out handle must be GPU");
        assert!(grad_input.is_gpu(), "grad_input handle must be GPU");
        assert!(grad_lambda.is_gpu(), "grad_lambda view must point to GPU buffer");
        assert!(lambda.is_gpu(), "Lambda view must point to GPU buffer");
        assert_eq!(lambda.len(), 1, "Lambda must be a single scalar");
        assert_eq!(grad_lambda.len(), 1, "grad_lambda must be a single element");

        let batch = input.rows();
        let features = input.cols();
        let total = batch * features;
        assert_eq!(grad_out.rows(), batch);
        assert_eq!(grad_out.cols(), features);
        assert_eq!(grad_input.rows(), batch);
        assert_eq!(grad_input.cols(), features);

        let in_buf = self.get_gpu_subbuffer_from_handle(input);
        let go_buf = self.get_gpu_subbuffer_from_handle(grad_out);
        let lambda_buf = subbuffer_from_view(self, lambda);
        let gi_buf = self.get_gpu_subbuffer_from_handle(grad_input);
        let grad_lambda_buf = subbuffer_from_view(self, grad_lambda);

        let pipeline = &self.learnable_mish_pipelines().backward;
        let push = [total as u32];
        self.run_compute_shader(
            pipeline,
            &[
                (0, in_buf),
                (1, go_buf),
                (2, lambda_buf),
                (3, gi_buf),
                (4, grad_lambda_buf),
            ],
            &push,
            total,
        );
    }
}