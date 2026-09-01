pub mod pipeline;   // <-- новый модуль

use crate::compute_manager::gpu::compute::GpuCompute;
use crate::compute_manager::matrix_buffer::MatrixBufferHandle;

impl GpuCompute {
    pub fn run_sigmoid_forward_buffered_handle(
        &self,
        input: &MatrixBufferHandle,
        output: &MatrixBufferHandle,
    ) {
        assert!(input.is_gpu() && output.is_gpu(), "Handles must be GPU");
        let total = input.rows() * input.cols();
        assert_eq!(total, output.rows() * output.cols(), "Shape mismatch");

        let in_buf = self.get_gpu_subbuffer_from_handle(input);
        let out_buf = self.get_gpu_subbuffer_from_handle(output);

        // Используем новый пайплайн из собственной структуры Sigmoid
        let pipeline = &self.sigmoid_pipelines().forward;
        let push = [total as u32];
        self.run_compute_shader(
            pipeline,
            &[(0, in_buf), (1, out_buf)],
            &push,
            total,
        );
    }

    pub fn run_sigmoid_backward_buffered_handle(
        &self,
        output: &MatrixBufferHandle,
        grad_output: &MatrixBufferHandle,
        grad_input: &MatrixBufferHandle,
    ) {
        assert!(output.is_gpu() && grad_output.is_gpu() && grad_input.is_gpu(), "Handles must be GPU");
        let total = output.rows() * output.cols();
        assert_eq!(total, grad_output.rows() * grad_output.cols(), "grad_output shape mismatch");
        assert_eq!(total, grad_input.rows() * grad_input.cols(), "grad_input shape mismatch");

        let y_buf = self.get_gpu_subbuffer_from_handle(output);
        let go_buf = self.get_gpu_subbuffer_from_handle(grad_output);
        let gi_buf = self.get_gpu_subbuffer_from_handle(grad_input);

        let pipeline = &self.sigmoid_pipelines().backward;
        let push = [total as u32];
        self.run_compute_shader(
            pipeline,
            &[(0, y_buf), (1, go_buf), (2, gi_buf)],
            &push,
            total,
        );
    }
}