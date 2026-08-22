// src/layers/identity/gpu/mod.rs

use crate::compute_manager::gpu::compute::GpuCompute;
use crate::compute_manager::matrix_buffer::MatrixBufferHandle;

impl GpuCompute {
    pub fn run_identity_forward_buffered_handle(
        &self,
        input: &MatrixBufferHandle,
        output: &MatrixBufferHandle,
    ) {
        assert!(input.is_gpu() && output.is_gpu(), "Handles must be GPU");
        let rows = input.rows();
        let cols = input.cols();
        assert_eq!(output.rows(), rows);
        assert_eq!(output.cols(), cols);
        self.copy_gpu_handle_to_gpu_handle(input, output);
    }

    pub fn run_identity_backward_buffered_handle(
        &self,
        grad_output: &MatrixBufferHandle,
        grad_input: &MatrixBufferHandle,
    ) {
        assert!(grad_output.is_gpu() && grad_input.is_gpu(), "Handles must be GPU");
        let rows = grad_output.rows();
        let cols = grad_output.cols();
        assert_eq!(grad_input.rows(), rows);
        assert_eq!(grad_input.cols(), cols);
        self.copy_gpu_handle_to_gpu_handle(grad_output, grad_input);
    }
}