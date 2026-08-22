// src/layers/tanh/gpu/mod.rs

use crate::compute_manager::gpu::compute::GpuCompute;
use crate::compute_manager::matrix_buffer::MatrixBufferHandle;

impl GpuCompute {
    pub fn run_tanh_forward_buffered_handle(
        &self,
        input: &MatrixBufferHandle,
        output: &MatrixBufferHandle,
    ) {
        self.run_activation_forward_buffered_handle(input, output, 2, 0.0)
    }

    pub fn run_tanh_backward_buffered_handle(
        &self,
        output: &MatrixBufferHandle,
        grad_output: &MatrixBufferHandle,
        grad_input: &MatrixBufferHandle,
    ) {
        self.run_activation_backward_buffered_handle(output, grad_output, grad_input, 2, 0.0)
    }
}