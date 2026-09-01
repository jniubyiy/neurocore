// src/losses/sub/gpu/mod.rs

pub mod pipeline;

use crate::compute_manager::gpu::compute::GpuCompute;
use crate::compute_manager::matrix_buffer::MatrixBufferHandle;

impl GpuCompute {
    pub fn run_sub_forward_buffered_handle(
        &self,
        pred: &MatrixBufferHandle,
        target: &MatrixBufferHandle,
        out: &MatrixBufferHandle,
    ) {
        assert!(pred.is_gpu() && target.is_gpu() && out.is_gpu(), "Handles must be GPU");
        let rows = pred.rows();
        let cols = pred.cols();
        let total = rows * cols;
        assert_eq!(target.rows(), rows);
        assert_eq!(target.cols(), cols);
        assert_eq!(out.rows(), rows);
        assert_eq!(out.cols(), cols);

        let a_buf = self.get_gpu_subbuffer_from_handle(pred);
        let b_buf = self.get_gpu_subbuffer_from_handle(target);
        let out_buf = self.get_gpu_subbuffer_from_handle(out);

        let pipeline = &self.sub_pipelines().forward;
        self.run_compute_shader(
            pipeline,
            &[(0, a_buf), (1, b_buf), (2, out_buf)],
            &[total as u32],
            total,
        );
    }

    pub fn run_sub_backward_buffered_handle(
        &self,
        grad_out: &MatrixBufferHandle,
        ga: &MatrixBufferHandle,
        gb: &MatrixBufferHandle,
    ) {
        assert!(grad_out.is_gpu() && ga.is_gpu() && gb.is_gpu(), "Handles must be GPU");
        let rows = grad_out.rows();
        let cols = grad_out.cols();
        let total = rows * cols;
        assert_eq!(ga.rows(), rows);
        assert_eq!(ga.cols(), cols);
        assert_eq!(gb.rows(), rows);
        assert_eq!(gb.cols(), cols);

        let go_buf = self.get_gpu_subbuffer_from_handle(grad_out);
        let ga_buf = self.get_gpu_subbuffer_from_handle(ga);
        let gb_buf = self.get_gpu_subbuffer_from_handle(gb);

        let pipeline = &self.sub_pipelines().backward;
        self.run_compute_shader(
            pipeline,
            &[(0, go_buf), (1, ga_buf), (2, gb_buf)],
            &[total as u32],
            total,
        );
    }
}
