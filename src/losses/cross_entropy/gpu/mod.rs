// src/losses/cross_entropy/gpu/mod.rs

pub mod pipeline;

use crate::compute_manager::gpu::compute::GpuCompute;
use crate::compute_manager::matrix_buffer::MatrixBufferHandle;

impl GpuCompute {
    pub fn run_cross_entropy_forward_buffered_handle(
        &self,
        logits_and_target: &MatrixBufferHandle,
        num_classes: usize,
        out: &MatrixBufferHandle,
    ) {
        assert!(logits_and_target.is_gpu() && out.is_gpu(), "Handles must be GPU");
        let batch = logits_and_target.rows();
        let cols = logits_and_target.cols();
        assert_eq!(cols, num_classes + 1);
        assert_eq!(out.rows(), batch);
        assert_eq!(out.cols(), 1);

        let in_buf = self.get_gpu_subbuffer_from_handle(logits_and_target);
        let out_buf = self.get_gpu_subbuffer_from_handle(out);

        let pipeline = self.cross_entropy_pipelines().forward.clone();
        self.run_compute_shader_with_dispatch(
            pipeline,
            &[(0, in_buf), (1, out_buf)],
            &[batch as u32, num_classes as u32],
            [batch as u32, 1, 1],
        );
    }

    pub fn run_cross_entropy_backward_buffered_handle(
        &self,
        logits_and_target: &MatrixBufferHandle,
        grad_out: &MatrixBufferHandle,
        num_classes: usize,
        gi: &MatrixBufferHandle,
    ) {
        assert!(logits_and_target.is_gpu() && grad_out.is_gpu() && gi.is_gpu(), "Handles must be GPU");
        let batch = logits_and_target.rows();
        let cols = logits_and_target.cols();
        assert_eq!(cols, num_classes + 1);
        assert_eq!(grad_out.rows(), batch);
        assert_eq!(grad_out.cols(), 1);
        assert_eq!(gi.rows(), batch);
        assert_eq!(gi.cols(), cols);

        let in_buf = self.get_gpu_subbuffer_from_handle(logits_and_target);
        let go_buf = self.get_gpu_subbuffer_from_handle(grad_out);
        let gi_buf = self.get_gpu_subbuffer_from_handle(gi);

        let pipeline = self.cross_entropy_pipelines().backward.clone();
        self.run_compute_shader_with_dispatch(
            pipeline,
            &[(0, in_buf), (1, go_buf), (2, gi_buf)],
            &[batch as u32, num_classes as u32],
            [batch as u32, 1, 1],
        );
    }
}
