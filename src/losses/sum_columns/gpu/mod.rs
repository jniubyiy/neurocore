// src/losses/sum_columns/gpu/mod.rs

pub mod pipeline;

use crate::compute_manager::gpu::compute::GpuCompute;
use crate::compute_manager::matrix_buffer::MatrixBufferHandle;

impl GpuCompute {
    pub fn run_sum_columns_forward_buffered_handle(
        &self,
        input: &MatrixBufferHandle,
        out: &MatrixBufferHandle,
    ) {
        assert!(input.is_gpu() && out.is_gpu(), "Handles must be GPU");
        let rows = input.rows();
        let cols = input.cols();
        assert_eq!(out.rows(), rows);
        assert_eq!(out.cols(), 1);

        let in_buf = self.get_gpu_subbuffer_from_handle(input);
        let out_buf = self.get_gpu_subbuffer_from_handle(out);

        let pipeline = self.sum_columns_pipelines().forward.clone();
        let push = [rows as u32, cols as u32];
        self.run_compute_shader_with_dispatch(
            pipeline,
            &[(0, in_buf), (1, out_buf)],
            &push,
            [rows as u32, 1, 1],
        );
    }

    pub fn run_sum_columns_backward_buffered_handle(
        &self,
        grad_out: &MatrixBufferHandle,
        original_cols: usize,
        gi: &MatrixBufferHandle,
    ) {
        assert!(grad_out.is_gpu() && gi.is_gpu(), "Handles must be GPU");
        let rows = grad_out.rows();
        assert_eq!(gi.rows(), rows);
        assert_eq!(gi.cols(), original_cols);

        let go_buf = self.get_gpu_subbuffer_from_handle(grad_out);
        let gi_buf = self.get_gpu_subbuffer_from_handle(gi);

        let pipeline = self.sum_columns_pipelines().backward.clone();
        let push = [rows as u32, original_cols as u32];
        self.run_compute_shader_with_dispatch(
            pipeline,
            &[(0, go_buf), (1, gi_buf)],
            &push,
            [rows as u32, original_cols as u32, 1],
        );
    }
}
