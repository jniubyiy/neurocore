// src/compute_manager/gpu/compute/loss_cubes.rs

use super::base::GpuCompute;
use crate::compute_manager::matrix_buffer::MatrixBufferHandle;

impl GpuCompute {
    // ===================================================================
    // Sub
    // ===================================================================

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

        let pipeline = self.pipeline_cache.sub_fwd.clone();
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

        let pipeline = self.pipeline_cache.sub_bwd.clone();
        self.run_compute_shader(
            pipeline,
            &[(0, go_buf), (1, ga_buf), (2, gb_buf)],
            &[total as u32],
            total,
        );
    }

    // ===================================================================
    // Square
    // ===================================================================

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

        self.run_compute_shader(
            self.pipeline_cache.square_fwd.clone(),
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

        self.run_compute_shader(
            self.pipeline_cache.square_bwd.clone(),
            &[(0, in_buf), (1, go_buf), (2, gi_buf)],
            &[total as u32],
            total,
        );
    }

    // ===================================================================
    // Abs
    // ===================================================================

    pub fn run_abs_forward_buffered_handle(
        &self,
        input: &MatrixBufferHandle,
        out: &MatrixBufferHandle,
    ) {
        assert!(input.is_gpu() && out.is_gpu(), "Handles must be GPU");
        let total = input.rows() * input.cols();
        assert_eq!(total, out.rows() * out.cols(), "Shape mismatch");

        let in_buf = self.get_gpu_subbuffer_from_handle(input);
        let out_buf = self.get_gpu_subbuffer_from_handle(out);

        self.run_compute_shader(
            self.pipeline_cache.abs_fwd.clone(),
            &[(0, in_buf), (1, out_buf)],
            &[total as u32],
            total,
        );
    }

    pub fn run_abs_backward_buffered_handle(
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

        self.run_compute_shader(
            self.pipeline_cache.abs_bwd.clone(),
            &[(0, in_buf), (1, go_buf), (2, gi_buf)],
            &[total as u32],
            total,
        );
    }

    // ===================================================================
    // Log1p
    // ===================================================================

    pub fn run_log1p_forward_buffered_handle(
        &self,
        input: &MatrixBufferHandle,
        out: &MatrixBufferHandle,
    ) {
        assert!(input.is_gpu() && out.is_gpu(), "Handles must be GPU");
        let total = input.rows() * input.cols();
        assert_eq!(total, out.rows() * out.cols(), "Shape mismatch");

        let in_buf = self.get_gpu_subbuffer_from_handle(input);
        let out_buf = self.get_gpu_subbuffer_from_handle(out);

        self.run_compute_shader(
            self.pipeline_cache.log1p_fwd.clone(),
            &[(0, in_buf), (1, out_buf)],
            &[total as u32],
            total,
        );
    }

    pub fn run_log1p_backward_buffered_handle(
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

        self.run_compute_shader(
            self.pipeline_cache.log1p_bwd.clone(),
            &[(0, in_buf), (1, go_buf), (2, gi_buf)],
            &[total as u32],
            total,
        );
    }

    // ===================================================================
    // AbsDiff
    // ===================================================================

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

        self.run_compute_shader(
            self.pipeline_cache.absdiff_fwd.clone(),
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

        self.run_compute_shader(
            self.pipeline_cache.absdiff_bwd.clone(),
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

    // ===================================================================
    // Log
    // ===================================================================

    pub fn run_log_forward_buffered_handle(
        &self,
        input: &MatrixBufferHandle,
        out: &MatrixBufferHandle,
    ) {
        assert!(input.is_gpu() && out.is_gpu(), "Handles must be GPU");
        let total = input.rows() * input.cols();
        assert_eq!(total, out.rows() * out.cols(), "Shape mismatch");

        let in_buf = self.get_gpu_subbuffer_from_handle(input);
        let out_buf = self.get_gpu_subbuffer_from_handle(out);

        self.run_compute_shader(
            self.pipeline_cache.log_fwd.clone(),
            &[(0, in_buf), (1, out_buf)],
            &[total as u32],
            total,
        );
    }

    pub fn run_log_backward_buffered_handle(
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

        self.run_compute_shader(
            self.pipeline_cache.log_bwd.clone(),
            &[(0, in_buf), (1, go_buf), (2, gi_buf)],
            &[total as u32],
            total,
        );
    }

    // ===================================================================
    // Neg
    // ===================================================================

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

        self.run_compute_shader(
            self.pipeline_cache.neg_fwd.clone(),
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

        self.run_compute_shader(
            self.pipeline_cache.neg_bwd.clone(),
            &[(0, go_buf), (1, gi_buf)],
            &[total as u32],
            total,
        );
    }

    // ===================================================================
    // Mul
    // ===================================================================

    pub fn run_mul_forward_buffered_handle(
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

        self.run_compute_shader(
            self.pipeline_cache.mul_fwd.clone(),
            &[(0, a_buf), (1, b_buf), (2, out_buf)],
            &[total as u32],
            total,
        );
    }

    pub fn run_mul_backward_buffered_handle(
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

        self.run_compute_shader(
            self.pipeline_cache.mul_bwd.clone(),
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

    // ===================================================================
    // AddScalar
    // ===================================================================

    pub fn run_addscalar_forward_buffered_handle(
        &self,
        input: &MatrixBufferHandle,
        scalar: f32,
        out: &MatrixBufferHandle,
    ) {
        assert!(input.is_gpu() && out.is_gpu(), "Handles must be GPU");
        let total = input.rows() * input.cols();
        assert_eq!(total, out.rows() * out.cols(), "Shape mismatch");

        let in_buf = self.get_gpu_subbuffer_from_handle(input);
        let out_buf = self.get_gpu_subbuffer_from_handle(out);

        self.run_compute_shader(
            self.pipeline_cache.addscalar_fwd.clone(),
            &[(0, in_buf), (1, out_buf)],
            &[total as u32, scalar.to_bits()],
            total,
        );
    }

    pub fn run_addscalar_backward_buffered_handle(
        &self,
        grad_out: &MatrixBufferHandle,
        gi: &MatrixBufferHandle,
    ) {
        assert!(grad_out.is_gpu() && gi.is_gpu(), "Handles must be GPU");
        let total = grad_out.rows() * grad_out.cols();
        assert_eq!(total, gi.rows() * gi.cols(), "Shape mismatch");

        let go_buf = self.get_gpu_subbuffer_from_handle(grad_out);
        let gi_buf = self.get_gpu_subbuffer_from_handle(gi);

        self.run_compute_shader(
            self.pipeline_cache.addscalar_bwd.clone(),
            &[(0, go_buf), (1, gi_buf)],
            &[total as u32],
            total,
        );
    }

    // ===================================================================
    // CrossEntropy
    // ===================================================================

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

        self.run_compute_shader_with_dispatch(
            self.pipeline_cache.cross_entropy_fwd.clone(),
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

        self.run_compute_shader_with_dispatch(
            self.pipeline_cache.cross_entropy_bwd.clone(),
            &[(0, in_buf), (1, go_buf), (2, gi_buf)],
            &[batch as u32, num_classes as u32],
            [batch as u32, 1, 1],
        );
    }

    // ===================================================================
    // SumColumns
    // ===================================================================

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

        let pipeline = self.pipeline_cache.reduce_pipeline();
        let push = [rows as u32];

        self.run_compute_shader_with_dispatch(
            pipeline,
            &[(0, in_buf), (1, out_buf)],
            &push,
            [cols as u32, 1, 1],
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

        let grad_vec = self.download_gpu_handle_to_vec(grad_out);

        let mut broadcast_vec = Vec::with_capacity(rows * original_cols);
        for _ in 0..original_cols {
            broadcast_vec.extend_from_slice(&grad_vec);
        }

        self.copy_slice_to_gpu_handle(gi, &broadcast_vec);
    }
}