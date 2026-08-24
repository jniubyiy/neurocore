pub mod pipeline;   // <-- новый модуль

use crate::compute_manager::gpu::compute::GpuCompute;
use crate::compute_manager::matrix_buffer::MatrixBufferHandle;

impl GpuCompute {
    pub fn run_leaky_relu_forward_buffered_handle(
        &self,
        input: &MatrixBufferHandle,
        output: &MatrixBufferHandle,
        alpha: f32,
    ) {
        assert!(input.is_gpu() && output.is_gpu(), "Handles must be GPU");
        let total = input.rows() * input.cols();
        assert_eq!(total, output.rows() * output.cols(), "Shape mismatch");

        let in_buf = self.get_gpu_subbuffer_from_handle(input);
        let out_buf = self.get_gpu_subbuffer_from_handle(output);

        // Используем новый пайплайн из собственной структуры LeakyReLU
        let pipeline = self.leaky_relu_pipelines().forward.clone();
        let push = [total as u32, alpha.to_bits()];
        self.run_compute_shader(
            pipeline,
            &[(0, in_buf), (1, out_buf)],
            &push,
            total,
        );
    }

    pub fn run_leaky_relu_backward_buffered_handle(
        &self,
        input: &MatrixBufferHandle,
        grad_output: &MatrixBufferHandle,
        grad_input: &MatrixBufferHandle,
        alpha: f32,
    ) {
        assert!(input.is_gpu() && grad_output.is_gpu() && grad_input.is_gpu(), "Handles must be GPU");
        let total = input.rows() * input.cols();
        assert_eq!(total, grad_output.rows() * grad_output.cols(), "grad_output shape mismatch");
        assert_eq!(total, grad_input.rows() * grad_input.cols(), "grad_input shape mismatch");

        let in_buf = self.get_gpu_subbuffer_from_handle(input);
        let go_buf = self.get_gpu_subbuffer_from_handle(grad_output);
        let gi_buf = self.get_gpu_subbuffer_from_handle(grad_input);

        let pipeline = self.leaky_relu_pipelines().backward.clone();
        let push = [total as u32, alpha.to_bits()];
        self.run_compute_shader(
            pipeline,
            &[(0, in_buf), (1, go_buf), (2, gi_buf)],
            &push,
            total,
        );
    }
}