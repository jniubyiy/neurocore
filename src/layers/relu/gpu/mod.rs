// src/layers/relu/gpu/mod.rs

use crate::compute_manager::gpu::compute::GpuCompute;
use crate::compute_manager::matrix_buffer::MatrixBufferHandle;

impl GpuCompute {
    pub fn run_relu_forward_buffered_handle(
        &self,
        input: &MatrixBufferHandle,
        output: &MatrixBufferHandle,
    ) {
        self.run_activation_forward_buffered_handle(input, output, 0, 0.0)
    }

    pub fn run_relu_backward_buffered_handle(
        &self,
        input: &MatrixBufferHandle,
        grad_output: &MatrixBufferHandle,
        grad_input: &MatrixBufferHandle,
    ) {
        self.run_activation_backward_buffered_handle(input, grad_output, grad_input, 0, 0.0)
    }
}