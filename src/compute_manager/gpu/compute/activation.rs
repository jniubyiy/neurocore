// src/compute_manager/gpu/compute/activation.rs

use super::base::GpuCompute;
use crate::compute_manager::matrix_buffer::MatrixBufferHandle;

impl GpuCompute {
    // ===================================================================
    // Handle-версии (MatrixBufferHandle)
    // ===================================================================

    /// Прямой проход активации на GPU с использованием MatrixBufferHandle.
    /// Вход и выход должны быть GPU-буферами.
    pub fn run_activation_forward_buffered_handle(
        &self,
        input: &MatrixBufferHandle,
        output: &MatrixBufferHandle,
        op_type: u32,
        alpha: f32,
    ) {
        assert!(input.is_gpu() && output.is_gpu(), "Handles must be GPU");
        let total_elements = input.rows() * input.cols();
        assert_eq!(total_elements, output.rows() * output.cols(), "Shape mismatch");

        let in_buf = self.get_gpu_subbuffer_from_handle(input);
        let out_buf = self.get_gpu_subbuffer_from_handle(output);

        let pipeline = self.pipeline_cache.activation_pipeline();
        let push: [u32; 3] = [op_type, alpha.to_bits(), total_elements as u32];

        self.run_compute_shader(
            pipeline,
            &[(0, in_buf), (1, out_buf)],
            &push,
            total_elements,
        );
    }

    /// Обратный проход активации на GPU с использованием MatrixBufferHandle.
    /// Вход/выход (в зависимости от операции) и градиенты должны быть GPU-буферами.
    pub fn run_activation_backward_buffered_handle(
        &self,
        input_or_output: &MatrixBufferHandle,
        grad_out: &MatrixBufferHandle,
        grad_input: &MatrixBufferHandle,
        op_type: u32,
        alpha: f32,
    ) {
        assert!(input_or_output.is_gpu(), "input_or_output must be GPU");
        assert!(grad_out.is_gpu(), "grad_out must be GPU");
        assert!(grad_input.is_gpu(), "grad_input must be GPU");

        let total_elements = input_or_output.rows() * input_or_output.cols();
        assert_eq!(total_elements, grad_out.rows() * grad_out.cols(), "grad_out shape mismatch");
        assert_eq!(total_elements, grad_input.rows() * grad_input.cols(), "grad_input shape mismatch");

        let in_buf = self.get_gpu_subbuffer_from_handle(input_or_output);
        let go_buf = self.get_gpu_subbuffer_from_handle(grad_out);
        let gi_buf = self.get_gpu_subbuffer_from_handle(grad_input);

        let pipeline = self.pipeline_cache.activation_backward_pipeline();
        let push: [u32; 3] = [op_type, alpha.to_bits(), total_elements as u32];

        self.run_compute_shader(
            pipeline,
            &[(0, in_buf), (1, go_buf), (2, gi_buf)],
            &push,
            total_elements,
        );
    }
}