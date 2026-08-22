// src/layers/leaky_relu/gpu/mod.rs

use crate::compute_manager::gpu::compute::GpuCompute;
use crate::compute_manager::matrix_buffer::MatrixBufferHandle;

impl GpuCompute {
    pub fn run_leaky_relu_forward_buffered_handle(
        &self,
        input: &MatrixBufferHandle,
        output: &MatrixBufferHandle,
        alpha: f32,
    ) {
        self.run_activation_forward_buffered_handle(input, output, 3, alpha)
    }

    pub fn run_leaky_relu_backward_buffered_handle(
        &self,
        input: &MatrixBufferHandle,
        grad_output: &MatrixBufferHandle,
        grad_input: &MatrixBufferHandle,
        alpha: f32,
    ) {
        self.run_activation_backward_buffered_handle(input, grad_output, grad_input, 3, alpha)
    }
}