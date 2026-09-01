// src/losses/neg/gpu/mod.rs

pub mod pipeline;

use crate::compute_manager::gpu::compute::GpuCompute;
use crate::compute_manager::matrix_buffer::MatrixBufferHandle;

impl GpuCompute {
    pub fn run_neg_forward_buffered_handle(
        &self,
        input: &MatrixBufferHandle,
        out: &MatrixBufferHandle,
    ) {
        assert!(input.is_gpu() && out.is_gpu(), "Handles must be GPU");
        let total = input.rows() * input.cols();
        assert_eq!(total, out.rows() * out.cols(), "Shape mismatch");

        let in_buf = self.get_gpu_subbuffer_from_handle(input);
        let out_buf = self.get_gpu_subbuffer_from_handle(out);

        let pipeline = &self.neg_pipelines().forward;
        self.run_compute_shader(
            pipeline,
            &[(0, in_buf), (1, out_buf)],
            &[total as u32],
            total,
        );
    }

    pub fn run_neg_backward_buffered_handle(
        &self,
        grad_out: &MatrixBufferHandle,
        gi: &MatrixBufferHandle,
    ) {
        assert!(grad_out.is_gpu() && gi.is_gpu(), "Handles must be GPU");
        let total = grad_out.rows() * grad_out.cols();
        assert_eq!(total, gi.rows() * gi.cols(), "Shape mismatch");

        let go_buf = self.get_gpu_subbuffer_from_handle(grad_out);
        let gi_buf = self.get_gpu_subbuffer_from_handle(gi);

        let pipeline = &self.neg_pipelines().backward;
        self.run_compute_shader(
            pipeline,
            &[(0, go_buf), (1, gi_buf)],
            &[total as u32],
            total,
        );
    }
}
