// src/losses/abs_diff/gpu/mod.rs

pub mod pipeline;

use crate::compute_manager::gpu::compute::GpuCompute;
use crate::compute_manager::matrix_buffer::MatrixBufferHandle;

impl GpuCompute {
    pub fn run_absdiff_forward_buffered_handle(
        &self,
        a: &MatrixBufferHandle,
        b: &MatrixBufferHandle,
        out: &MatrixBufferHandle,
    ) {
        assert!(a.is_gpu() && b.is_gpu() && out.is_gpu(), "Handles must be GPU");
        let total = a.rows() * a.cols();
        assert_eq!(b.rows() * b.cols(), total, "b shape mismatch");
        assert_eq!(out.rows() * out.cols(), total, "out shape mismatch");

        let a_buf = self.get_gpu_subbuffer_from_handle(a);
        let b_buf = self.get_gpu_subbuffer_from_handle(b);
        let out_buf = self.get_gpu_subbuffer_from_handle(out);

        let pipeline = self.abs_diff_pipelines().forward.clone();
        self.run_compute_shader(
            pipeline,
            &[(0, a_buf), (1, b_buf), (2, out_buf)],
            &[total as u32],
            total,
        );
    }

    pub fn run_absdiff_backward_buffered_handle(
        &self,
        a: &MatrixBufferHandle,
        b: &MatrixBufferHandle,
        grad_out: &MatrixBufferHandle,
        ga: &MatrixBufferHandle,
        gb: &MatrixBufferHandle,
    ) {
        assert!(a.is_gpu() && b.is_gpu() && grad_out.is_gpu() && ga.is_gpu() && gb.is_gpu(), "Handles must be GPU");
        let total = a.rows() * a.cols();
        assert_eq!(b.rows() * b.cols(), total, "b shape mismatch");
        assert_eq!(grad_out.rows() * grad_out.cols(), total, "grad_out shape mismatch");
        assert_eq!(ga.rows() * ga.cols(), total, "ga shape mismatch");
        assert_eq!(gb.rows() * gb.cols(), total, "gb shape mismatch");

        let a_buf = self.get_gpu_subbuffer_from_handle(a);
        let b_buf = self.get_gpu_subbuffer_from_handle(b);
        let go_buf = self.get_gpu_subbuffer_from_handle(grad_out);
        let ga_buf = self.get_gpu_subbuffer_from_handle(ga);
        let gb_buf = self.get_gpu_subbuffer_from_handle(gb);

        let pipeline = self.abs_diff_pipelines().backward.clone();
        self.run_compute_shader(
            pipeline,
            &[
                (0, a_buf),
                (1, b_buf),
                (2, go_buf),
                (3, ga_buf),
                (4, gb_buf),
            ],
            &[total as u32],
            total,
        );
    }
}
