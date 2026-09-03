// src/layers/linear_attention/gpu/mod.rs

pub mod pipeline;

use crate::compute_manager::gpu::compute::GpuCompute;
use crate::compute_manager::matrix_buffer::MatrixBufferHandle;

impl GpuCompute {
    pub fn run_linear_attention_forward_buffered_handle(
        &self,
        _input: &MatrixBufferHandle,
        _params: &MatrixBufferHandle,
        _output: &MatrixBufferHandle,
    ) {
        panic!("LinearAttention GPU forward is not implemented");
    }

    pub fn run_linear_attention_backward_buffered_handle(
        &self,
        _input: &MatrixBufferHandle,
        _grad_out: &MatrixBufferHandle,
        _params: &MatrixBufferHandle,
        _grad_input: &MatrixBufferHandle,
        _grad_params: &MatrixBufferHandle,
    ) {
        panic!("LinearAttention GPU backward is not implemented");
    }
}