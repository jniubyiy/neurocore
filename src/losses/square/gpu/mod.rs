// src/losses/square/gpu/mod.rs

pub mod pipeline;

use crate::compute_manager::gpu::compute::GpuCompute;
use crate::compute_manager::matrix_buffer::MatrixBufferHandle;

impl GpuCompute {
    pub fn run_square_forward_buffered_handle(
        &self,
        input: &MatrixBufferHandle,
        out: &MatrixBufferHandle,
    ) {
        assert!(input.is_gpu() && out.is_gpu(), "Handles must be GPU");
        let total = input.rows() * input.cols();
        assert_eq!(total, out.rows() * out.cols(), "Shape mismatch");

        let in_buf = self.get_gpu_subbuffer_from_handle(input);
        let out_buf = self.get_gpu_subbuffer_from_handle(out);

        let pipeline = &self.square_pipelines().forward;
        self.run_compute_shader(
            pipeline,
            &[(0, in_buf), (1, out_buf)],
            &[total as u32],
            total,
        );
    }

    pub fn run_square_backward_buffered_handle(
        &self,
        input: &MatrixBufferHandle,
        grad_out: &MatrixBufferHandle,
        gi: &MatrixBufferHandle,
    ) {
        assert!(input.is_gpu() && grad_out.is_gpu() && gi.is_gpu(), "Handles must be GPU");
        let total = input.rows() * input.cols();
        assert_eq!(total, grad_out.rows() * grad_out.cols(), "grad_out shape mismatch");
        assert_eq!(total, gi.rows() * gi.cols(), "gi shape mismatch");

        let in_buf = self.get_gpu_subbuffer_from_handle(input);
        let go_buf = self.get_gpu_subbuffer_from_handle(grad_out);
        let gi_buf = self.get_gpu_subbuffer_from_handle(gi);

        let pipeline = &self.square_pipelines().backward;
        self.run_compute_shader(
            pipeline,
            &[(0, in_buf), (1, go_buf), (2, gi_buf)],
            &[total as u32],
            total,
        );
    }
}
