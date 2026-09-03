// src/layers/ind_rnn/gpu/mod.rs

pub mod pipeline;

use crate::compute_manager::gpu::compute::GpuCompute;
use crate::compute_manager::matrix_buffer::MatrixBufferHandle;

impl GpuCompute {
    pub fn run_ind_rnn_forward_buffered_handle(
        &self,
        _input: &MatrixBufferHandle,
        _params: &MatrixBufferHandle,
        _output: &MatrixBufferHandle,
    ) {
        panic!("IndRNN GPU forward is not implemented");
    }

    pub fn run_ind_rnn_backward_buffered_handle(
        &self,
        _input: &MatrixBufferHandle,
        _grad_out: &MatrixBufferHandle,
        _params: &MatrixBufferHandle,
        _grad_input: &MatrixBufferHandle,
        _grad_params: &MatrixBufferHandle,
    ) {
        panic!("IndRNN GPU backward is not implemented");
    }
}