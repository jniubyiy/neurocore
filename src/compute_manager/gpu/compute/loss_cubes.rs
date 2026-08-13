// src/compute_manager/gpu/compute/loss_cubes.rs

use faer::Mat;
use vulkano::buffer::Subbuffer;
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::{Pipeline, PipelineBindPoint};
use super::base::GpuCompute;
use crate::compute_manager::matrix_buffer::MatrixBuffer;

impl GpuCompute {
    // ===================================================================
    // Sub
    // ===================================================================

    /// Старая версия для обратной совместимости.
    pub fn run_sub_forward(&self, pred: &Mat<f32>, target: &Mat<f32>) -> Mat<f32> {
        let total = pred.nrows() * pred.ncols();
        let (a_buf, a_raw) = self.upload_to_temp_buffer(&Self::mat_to_flat(pred));
        let (b_buf, b_raw) = self.upload_to_temp_buffer(&Self::mat_to_flat(target));
        let (out_buf, out_raw) = self.acquire_temp_buffer(total);

        let pipeline = self.pipeline_cache.sub_fwd.clone();
        self.run_compute_shader(
            pipeline,
            &[(0, a_buf.clone()), (1, b_buf.clone()), (2, out_buf.clone())],
            &[total as u32],
            total,
        );
        let mat = self.read_temp_buffer_to_mat(out_buf, out_raw, pred.nrows(), pred.ncols());
        self.release_temp_buffer(a_buf, a_raw);
        self.release_temp_buffer(b_buf, b_raw);
        mat
    }

    pub fn run_sub_backward(&self, grad_out: &Mat<f32>) -> (Mat<f32>, Mat<f32>) {
        let total = grad_out.nrows() * grad_out.ncols();
        let (go_buf, go_raw) = self.upload_to_temp_buffer(&Self::mat_to_flat(grad_out));
        let (ga_buf, ga_raw) = self.acquire_temp_buffer(total);
        let (gb_buf, gb_raw) = self.acquire_temp_buffer(total);

        let pipeline = self.pipeline_cache.sub_bwd.clone();
        self.run_compute_shader(
            pipeline,
            &[(0, go_buf.clone()), (1, ga_buf.clone()), (2, gb_buf.clone())],
            &[total as u32],
            total,
        );
        let ga = self.read_temp_buffer_to_mat(ga_buf, ga_raw, grad_out.nrows(), grad_out.ncols());
        let gb = self.read_temp_buffer_to_mat(gb_buf, gb_raw, grad_out.nrows(), grad_out.ncols());
        self.release_temp_buffer(go_buf, go_raw);
        (ga, gb)
    }

    /// Буферизованная версия: принимает GPU-буферы, возвращает GPU-буфер.
    pub fn run_sub_forward_buffered(&self, pred: &MatrixBuffer, target: &MatrixBuffer) -> MatrixBuffer {
        assert!(pred.is_gpu() && target.is_gpu(), "Buffers must be GPU");
        let rows = pred.rows();
        let cols = pred.cols();
        let total = rows * cols;
        assert_eq!(target.rows(), rows);
        assert_eq!(target.cols(), cols);

        let a_buf = pred.as_gpu_buffer().expect("GPU buffer");
        let b_buf = target.as_gpu_buffer().expect("GPU buffer");
        let out = self.allocate_gpu_matrix(rows, cols);
        let out_buf = out.as_gpu_buffer().expect("GPU buffer");

        let pipeline = self.pipeline_cache.sub_fwd.clone();
        self.run_compute_shader(
            pipeline,
            &[(0, a_buf.clone()), (1, b_buf.clone()), (2, out_buf.clone())],
            &[total as u32],
            total,
        );
        out
    }

    pub fn run_sub_backward_buffered(&self, grad_out: &MatrixBuffer) -> (MatrixBuffer, MatrixBuffer) {
        assert!(grad_out.is_gpu(), "Buffer must be GPU");
        let rows = grad_out.rows();
        let cols = grad_out.cols();
        let total = rows * cols;

        let go_buf = grad_out.as_gpu_buffer().expect("GPU buffer");
        let ga = self.allocate_gpu_matrix(rows, cols);
        let gb = self.allocate_gpu_matrix(rows, cols);
        let ga_buf = ga.as_gpu_buffer().expect("GPU buffer");
        let gb_buf = gb.as_gpu_buffer().expect("GPU buffer");

        let pipeline = self.pipeline_cache.sub_bwd.clone();
        self.run_compute_shader(
            pipeline,
            &[(0, go_buf.clone()), (1, ga_buf.clone()), (2, gb_buf.clone())],
            &[total as u32],
            total,
        );
        (ga, gb)
    }

    // ===================================================================
    // Square
    // ===================================================================

    /// Старая версия.
    pub fn run_square_forward(&self, input: &Mat<f32>) -> Mat<f32> {
        let total = input.nrows() * input.ncols();
        let (in_buf, in_raw) = self.upload_to_temp_buffer(&Self::mat_to_flat(input));
        let (out_buf, out_raw) = self.acquire_temp_buffer(total);

        self.run_compute_shader(
            self.pipeline_cache.square_fwd.clone(),
            &[(0, in_buf.clone()), (1, out_buf.clone())],
            &[total as u32],
            total,
        );
        let mat = self.read_temp_buffer_to_mat(out_buf, out_raw, input.nrows(), input.ncols());
        self.release_temp_buffer(in_buf, in_raw);
        mat
    }

    pub fn run_square_backward(&self, input: &Mat<f32>, grad_out: &Mat<f32>) -> Mat<f32> {
        let total = input.nrows() * input.ncols();
        let (in_buf, in_raw) = self.upload_to_temp_buffer(&Self::mat_to_flat(input));
        let (go_buf, go_raw) = self.upload_to_temp_buffer(&Self::mat_to_flat(grad_out));
        let (gi_buf, gi_raw) = self.acquire_temp_buffer(total);

        self.run_compute_shader(
            self.pipeline_cache.square_bwd.clone(),
            &[(0, in_buf.clone()), (1, go_buf.clone()), (2, gi_buf.clone())],
            &[total as u32],
            total,
        );
        let mat = self.read_temp_buffer_to_mat(gi_buf, gi_raw, input.nrows(), input.ncols());
        self.release_temp_buffer(in_buf, in_raw);
        self.release_temp_buffer(go_buf, go_raw);
        mat
    }

    /// Буферизованные версии.
    pub fn run_square_forward_buffered(&self, input: &MatrixBuffer) -> MatrixBuffer {
        assert!(input.is_gpu(), "Buffer must be GPU");
        let rows = input.rows();
        let cols = input.cols();
        let total = rows * cols;

        let in_buf = input.as_gpu_buffer().expect("GPU buffer");
        let out = self.allocate_gpu_matrix(rows, cols);
        let out_buf = out.as_gpu_buffer().expect("GPU buffer");

        self.run_compute_shader(
            self.pipeline_cache.square_fwd.clone(),
            &[(0, in_buf.clone()), (1, out_buf.clone())],
            &[total as u32],
            total,
        );
        out
    }

    pub fn run_square_backward_buffered(&self, input: &MatrixBuffer, grad_out: &MatrixBuffer) -> MatrixBuffer {
        assert!(input.is_gpu() && grad_out.is_gpu(), "Buffers must be GPU");
        let rows = input.rows();
        let cols = input.cols();
        let total = rows * cols;
        assert_eq!(grad_out.rows(), rows);
        assert_eq!(grad_out.cols(), cols);

        let in_buf = input.as_gpu_buffer().expect("GPU buffer");
        let go_buf = grad_out.as_gpu_buffer().expect("GPU buffer");
        let gi = self.allocate_gpu_matrix(rows, cols);
        let gi_buf = gi.as_gpu_buffer().expect("GPU buffer");

        self.run_compute_shader(
            self.pipeline_cache.square_bwd.clone(),
            &[(0, in_buf.clone()), (1, go_buf.clone()), (2, gi_buf.clone())],
            &[total as u32],
            total,
        );
        gi
    }

    // ===================================================================
    // Abs
    // ===================================================================

    /// Старая версия.
    pub fn run_abs_forward(&self, input: &Mat<f32>) -> Mat<f32> {
        let total = input.nrows() * input.ncols();
        let (in_buf, in_raw) = self.upload_to_temp_buffer(&Self::mat_to_flat(input));
        let (out_buf, out_raw) = self.acquire_temp_buffer(total);

        self.run_compute_shader(
            self.pipeline_cache.abs_fwd.clone(),
            &[(0, in_buf.clone()), (1, out_buf.clone())],
            &[total as u32],
            total,
        );
        let mat = self.read_temp_buffer_to_mat(out_buf, out_raw, input.nrows(), input.ncols());
        self.release_temp_buffer(in_buf, in_raw);
        mat
    }

    pub fn run_abs_backward(&self, input: &Mat<f32>, grad_out: &Mat<f32>) -> Mat<f32> {
        let total = input.nrows() * input.ncols();
        let (in_buf, in_raw) = self.upload_to_temp_buffer(&Self::mat_to_flat(input));
        let (go_buf, go_raw) = self.upload_to_temp_buffer(&Self::mat_to_flat(grad_out));
        let (gi_buf, gi_raw) = self.acquire_temp_buffer(total);

        self.run_compute_shader(
            self.pipeline_cache.abs_bwd.clone(),
            &[(0, in_buf.clone()), (1, go_buf.clone()), (2, gi_buf.clone())],
            &[total as u32],
            total,
        );
        let mat = self.read_temp_buffer_to_mat(gi_buf, gi_raw, input.nrows(), input.ncols());
        self.release_temp_buffer(in_buf, in_raw);
        self.release_temp_buffer(go_buf, go_raw);
        mat
    }

    /// Буферизованные версии.
    pub fn run_abs_forward_buffered(&self, input: &MatrixBuffer) -> MatrixBuffer {
        assert!(input.is_gpu(), "Buffer must be GPU");
        let rows = input.rows();
        let cols = input.cols();
        let total = rows * cols;

        let in_buf = input.as_gpu_buffer().expect("GPU buffer");
        let out = self.allocate_gpu_matrix(rows, cols);
        let out_buf = out.as_gpu_buffer().expect("GPU buffer");

        self.run_compute_shader(
            self.pipeline_cache.abs_fwd.clone(),
            &[(0, in_buf.clone()), (1, out_buf.clone())],
            &[total as u32],
            total,
        );
        out
    }

    pub fn run_abs_backward_buffered(&self, input: &MatrixBuffer, grad_out: &MatrixBuffer) -> MatrixBuffer {
        assert!(input.is_gpu() && grad_out.is_gpu(), "Buffers must be GPU");
        let rows = input.rows();
        let cols = input.cols();
        let total = rows * cols;
        assert_eq!(grad_out.rows(), rows);
        assert_eq!(grad_out.cols(), cols);

        let in_buf = input.as_gpu_buffer().expect("GPU buffer");
        let go_buf = grad_out.as_gpu_buffer().expect("GPU buffer");
        let gi = self.allocate_gpu_matrix(rows, cols);
        let gi_buf = gi.as_gpu_buffer().expect("GPU buffer");

        self.run_compute_shader(
            self.pipeline_cache.abs_bwd.clone(),
            &[(0, in_buf.clone()), (1, go_buf.clone()), (2, gi_buf.clone())],
            &[total as u32],
            total,
        );
        gi
    }

    // ===================================================================
    // Log1p
    // ===================================================================

    /// Старая версия.
    pub fn run_log1p_forward(&self, input: &Mat<f32>) -> Mat<f32> {
        let total = input.nrows() * input.ncols();
        let (in_buf, in_raw) = self.upload_to_temp_buffer(&Self::mat_to_flat(input));
        let (out_buf, out_raw) = self.acquire_temp_buffer(total);

        self.run_compute_shader(
            self.pipeline_cache.log1p_fwd.clone(),
            &[(0, in_buf.clone()), (1, out_buf.clone())],
            &[total as u32],
            total,
        );
        let mat = self.read_temp_buffer_to_mat(out_buf, out_raw, input.nrows(), input.ncols());
        self.release_temp_buffer(in_buf, in_raw);
        mat
    }

    pub fn run_log1p_backward(&self, input: &Mat<f32>, grad_out: &Mat<f32>) -> Mat<f32> {
        let total = input.nrows() * input.ncols();
        let (in_buf, in_raw) = self.upload_to_temp_buffer(&Self::mat_to_flat(input));
        let (go_buf, go_raw) = self.upload_to_temp_buffer(&Self::mat_to_flat(grad_out));
        let (gi_buf, gi_raw) = self.acquire_temp_buffer(total);

        self.run_compute_shader(
            self.pipeline_cache.log1p_bwd.clone(),
            &[(0, in_buf.clone()), (1, go_buf.clone()), (2, gi_buf.clone())],
            &[total as u32],
            total,
        );
        let mat = self.read_temp_buffer_to_mat(gi_buf, gi_raw, input.nrows(), input.ncols());
        self.release_temp_buffer(in_buf, in_raw);
        self.release_temp_buffer(go_buf, go_raw);
        mat
    }

    /// Буферизованные версии.
    pub fn run_log1p_forward_buffered(&self, input: &MatrixBuffer) -> MatrixBuffer {
        assert!(input.is_gpu(), "Buffer must be GPU");
        let rows = input.rows();
        let cols = input.cols();
        let total = rows * cols;

        let in_buf = input.as_gpu_buffer().expect("GPU buffer");
        let out = self.allocate_gpu_matrix(rows, cols);
        let out_buf = out.as_gpu_buffer().expect("GPU buffer");

        self.run_compute_shader(
            self.pipeline_cache.log1p_fwd.clone(),
            &[(0, in_buf.clone()), (1, out_buf.clone())],
            &[total as u32],
            total,
        );
        out
    }

    pub fn run_log1p_backward_buffered(&self, input: &MatrixBuffer, grad_out: &MatrixBuffer) -> MatrixBuffer {
        assert!(input.is_gpu() && grad_out.is_gpu(), "Buffers must be GPU");
        let rows = input.rows();
        let cols = input.cols();
        let total = rows * cols;
        assert_eq!(grad_out.rows(), rows);
        assert_eq!(grad_out.cols(), cols);

        let in_buf = input.as_gpu_buffer().expect("GPU buffer");
        let go_buf = grad_out.as_gpu_buffer().expect("GPU buffer");
        let gi = self.allocate_gpu_matrix(rows, cols);
        let gi_buf = gi.as_gpu_buffer().expect("GPU buffer");

        self.run_compute_shader(
            self.pipeline_cache.log1p_bwd.clone(),
            &[(0, in_buf.clone()), (1, go_buf.clone()), (2, gi_buf.clone())],
            &[total as u32],
            total,
        );
        gi
    }

    // ===================================================================
    // AbsDiff
    // ===================================================================

    /// Старая версия.
    pub fn run_absdiff_forward(&self, a: &Mat<f32>, b: &Mat<f32>) -> Mat<f32> {
        let total = a.nrows() * a.ncols();
        let (a_buf, a_raw) = self.upload_to_temp_buffer(&Self::mat_to_flat(a));
        let (b_buf, b_raw) = self.upload_to_temp_buffer(&Self::mat_to_flat(b));
        let (out_buf, out_raw) = self.acquire_temp_buffer(total);

        self.run_compute_shader(
            self.pipeline_cache.absdiff_fwd.clone(),
            &[(0, a_buf.clone()), (1, b_buf.clone()), (2, out_buf.clone())],
            &[total as u32],
            total,
        );
        let mat = self.read_temp_buffer_to_mat(out_buf, out_raw, a.nrows(), a.ncols());
        self.release_temp_buffer(a_buf, a_raw);
        self.release_temp_buffer(b_buf, b_raw);
        mat
    }

    pub fn run_absdiff_backward(&self, a: &Mat<f32>, b: &Mat<f32>, grad_out: &Mat<f32>) -> (Mat<f32>, Mat<f32>) {
        let total = a.nrows() * a.ncols();
        let (a_buf, a_raw) = self.upload_to_temp_buffer(&Self::mat_to_flat(a));
        let (b_buf, b_raw) = self.upload_to_temp_buffer(&Self::mat_to_flat(b));
        let (go_buf, go_raw) = self.upload_to_temp_buffer(&Self::mat_to_flat(grad_out));
        let (ga_buf, ga_raw) = self.acquire_temp_buffer(total);
        let (gb_buf, gb_raw) = self.acquire_temp_buffer(total);

        self.run_compute_shader(
            self.pipeline_cache.absdiff_bwd.clone(),
            &[
                (0, a_buf.clone()),
                (1, b_buf.clone()),
                (2, go_buf.clone()),
                (3, ga_buf.clone()),
                (4, gb_buf.clone()),
            ],
            &[total as u32],
            total,
        );
        let ga = self.read_temp_buffer_to_mat(ga_buf, ga_raw, a.nrows(), a.ncols());
        let gb = self.read_temp_buffer_to_mat(gb_buf, gb_raw, a.nrows(), a.ncols());
        self.release_temp_buffer(a_buf, a_raw);
        self.release_temp_buffer(b_buf, b_raw);
        self.release_temp_buffer(go_buf, go_raw);
        (ga, gb)
    }

    /// Буферизованные версии.
    pub fn run_absdiff_forward_buffered(&self, a: &MatrixBuffer, b: &MatrixBuffer) -> MatrixBuffer {
        assert!(a.is_gpu() && b.is_gpu(), "Buffers must be GPU");
        let rows = a.rows();
        let cols = a.cols();
        let total = rows * cols;
        assert_eq!(b.rows(), rows);
        assert_eq!(b.cols(), cols);

        let a_buf = a.as_gpu_buffer().expect("GPU buffer");
        let b_buf = b.as_gpu_buffer().expect("GPU buffer");
        let out = self.allocate_gpu_matrix(rows, cols);
        let out_buf = out.as_gpu_buffer().expect("GPU buffer");

        self.run_compute_shader(
            self.pipeline_cache.absdiff_fwd.clone(),
            &[(0, a_buf.clone()), (1, b_buf.clone()), (2, out_buf.clone())],
            &[total as u32],
            total,
        );
        out
    }

    pub fn run_absdiff_backward_buffered(&self, a: &MatrixBuffer, b: &MatrixBuffer, grad_out: &MatrixBuffer) -> (MatrixBuffer, MatrixBuffer) {
        assert!(a.is_gpu() && b.is_gpu() && grad_out.is_gpu(), "Buffers must be GPU");
        let rows = a.rows();
        let cols = a.cols();
        let total = rows * cols;
        assert_eq!(b.rows(), rows);
        assert_eq!(b.cols(), cols);
        assert_eq!(grad_out.rows(), rows);
        assert_eq!(grad_out.cols(), cols);

        let a_buf = a.as_gpu_buffer().expect("GPU buffer");
        let b_buf = b.as_gpu_buffer().expect("GPU buffer");
        let go_buf = grad_out.as_gpu_buffer().expect("GPU buffer");
        let ga = self.allocate_gpu_matrix(rows, cols);
        let gb = self.allocate_gpu_matrix(rows, cols);
        let ga_buf = ga.as_gpu_buffer().expect("GPU buffer");
        let gb_buf = gb.as_gpu_buffer().expect("GPU buffer");

        self.run_compute_shader(
            self.pipeline_cache.absdiff_bwd.clone(),
            &[
                (0, a_buf.clone()),
                (1, b_buf.clone()),
                (2, go_buf.clone()),
                (3, ga_buf.clone()),
                (4, gb_buf.clone()),
            ],
            &[total as u32],
            total,
        );
        (ga, gb)
    }

    // ===================================================================
    // Log
    // ===================================================================

    /// Старая версия.
    pub fn run_log_forward(&self, input: &Mat<f32>) -> Mat<f32> {
        let total = input.nrows() * input.ncols();
        let (in_buf, in_raw) = self.upload_to_temp_buffer(&Self::mat_to_flat(input));
        let (out_buf, out_raw) = self.acquire_temp_buffer(total);

        self.run_compute_shader(
            self.pipeline_cache.log_fwd.clone(),
            &[(0, in_buf.clone()), (1, out_buf.clone())],
            &[total as u32],
            total,
        );
        let mat = self.read_temp_buffer_to_mat(out_buf, out_raw, input.nrows(), input.ncols());
        self.release_temp_buffer(in_buf, in_raw);
        mat
    }

    pub fn run_log_backward(&self, input: &Mat<f32>, grad_out: &Mat<f32>) -> Mat<f32> {
        let total = input.nrows() * input.ncols();
        let (in_buf, in_raw) = self.upload_to_temp_buffer(&Self::mat_to_flat(input));
        let (go_buf, go_raw) = self.upload_to_temp_buffer(&Self::mat_to_flat(grad_out));
        let (gi_buf, gi_raw) = self.acquire_temp_buffer(total);

        self.run_compute_shader(
            self.pipeline_cache.log_bwd.clone(),
            &[(0, in_buf.clone()), (1, go_buf.clone()), (2, gi_buf.clone())],
            &[total as u32],
            total,
        );
        let mat = self.read_temp_buffer_to_mat(gi_buf, gi_raw, input.nrows(), input.ncols());
        self.release_temp_buffer(in_buf, in_raw);
        self.release_temp_buffer(go_buf, go_raw);
        mat
    }

    /// Буферизованные версии.
    pub fn run_log_forward_buffered(&self, input: &MatrixBuffer) -> MatrixBuffer {
        assert!(input.is_gpu(), "Buffer must be GPU");
        let rows = input.rows();
        let cols = input.cols();
        let total = rows * cols;

        let in_buf = input.as_gpu_buffer().expect("GPU buffer");
        let out = self.allocate_gpu_matrix(rows, cols);
        let out_buf = out.as_gpu_buffer().expect("GPU buffer");

        self.run_compute_shader(
            self.pipeline_cache.log_fwd.clone(),
            &[(0, in_buf.clone()), (1, out_buf.clone())],
            &[total as u32],
            total,
        );
        out
    }

    pub fn run_log_backward_buffered(&self, input: &MatrixBuffer, grad_out: &MatrixBuffer) -> MatrixBuffer {
        assert!(input.is_gpu() && grad_out.is_gpu(), "Buffers must be GPU");
        let rows = input.rows();
        let cols = input.cols();
        let total = rows * cols;
        assert_eq!(grad_out.rows(), rows);
        assert_eq!(grad_out.cols(), cols);

        let in_buf = input.as_gpu_buffer().expect("GPU buffer");
        let go_buf = grad_out.as_gpu_buffer().expect("GPU buffer");
        let gi = self.allocate_gpu_matrix(rows, cols);
        let gi_buf = gi.as_gpu_buffer().expect("GPU buffer");

        self.run_compute_shader(
            self.pipeline_cache.log_bwd.clone(),
            &[(0, in_buf.clone()), (1, go_buf.clone()), (2, gi_buf.clone())],
            &[total as u32],
            total,
        );
        gi
    }

    // ===================================================================
    // Neg
    // ===================================================================

    /// Старая версия.
    pub fn run_neg_forward(&self, input: &Mat<f32>) -> Mat<f32> {
        let total = input.nrows() * input.ncols();
        let (in_buf, in_raw) = self.upload_to_temp_buffer(&Self::mat_to_flat(input));
        let (out_buf, out_raw) = self.acquire_temp_buffer(total);

        self.run_compute_shader(
            self.pipeline_cache.neg_fwd.clone(),
            &[(0, in_buf.clone()), (1, out_buf.clone())],
            &[total as u32],
            total,
        );
        let mat = self.read_temp_buffer_to_mat(out_buf, out_raw, input.nrows(), input.ncols());
        self.release_temp_buffer(in_buf, in_raw);
        mat
    }

    pub fn run_neg_backward(&self, grad_out: &Mat<f32>) -> Mat<f32> {
        let total = grad_out.nrows() * grad_out.ncols();
        let (go_buf, go_raw) = self.upload_to_temp_buffer(&Self::mat_to_flat(grad_out));
        let (gi_buf, gi_raw) = self.acquire_temp_buffer(total);

        self.run_compute_shader(
            self.pipeline_cache.neg_bwd.clone(),
            &[(0, go_buf.clone()), (1, gi_buf.clone())],
            &[total as u32],
            total,
        );
        let mat = self.read_temp_buffer_to_mat(gi_buf, gi_raw, grad_out.nrows(), grad_out.ncols());
        self.release_temp_buffer(go_buf, go_raw);
        mat
    }

    /// Буферизованные версии.
    pub fn run_neg_forward_buffered(&self, input: &MatrixBuffer) -> MatrixBuffer {
        assert!(input.is_gpu(), "Buffer must be GPU");
        let rows = input.rows();
        let cols = input.cols();
        let total = rows * cols;

        let in_buf = input.as_gpu_buffer().expect("GPU buffer");
        let out = self.allocate_gpu_matrix(rows, cols);
        let out_buf = out.as_gpu_buffer().expect("GPU buffer");

        self.run_compute_shader(
            self.pipeline_cache.neg_fwd.clone(),
            &[(0, in_buf.clone()), (1, out_buf.clone())],
            &[total as u32],
            total,
        );
        out
    }

    pub fn run_neg_backward_buffered(&self, grad_out: &MatrixBuffer) -> MatrixBuffer {
        assert!(grad_out.is_gpu(), "Buffer must be GPU");
        let rows = grad_out.rows();
        let cols = grad_out.cols();
        let total = rows * cols;

        let go_buf = grad_out.as_gpu_buffer().expect("GPU buffer");
        let gi = self.allocate_gpu_matrix(rows, cols);
        let gi_buf = gi.as_gpu_buffer().expect("GPU buffer");

        self.run_compute_shader(
            self.pipeline_cache.neg_bwd.clone(),
            &[(0, go_buf.clone()), (1, gi_buf.clone())],
            &[total as u32],
            total,
        );
        gi
    }

    // ===================================================================
    // Mul
    // ===================================================================

    /// Старая версия.
    pub fn run_mul_forward(&self, a: &Mat<f32>, b: &Mat<f32>) -> Mat<f32> {
        let total = a.nrows() * a.ncols();
        let (a_buf, a_raw) = self.upload_to_temp_buffer(&Self::mat_to_flat(a));
        let (b_buf, b_raw) = self.upload_to_temp_buffer(&Self::mat_to_flat(b));
        let (out_buf, out_raw) = self.acquire_temp_buffer(total);

        self.run_compute_shader(
            self.pipeline_cache.mul_fwd.clone(),
            &[(0, a_buf.clone()), (1, b_buf.clone()), (2, out_buf.clone())],
            &[total as u32],
            total,
        );
        let mat = self.read_temp_buffer_to_mat(out_buf, out_raw, a.nrows(), a.ncols());
        self.release_temp_buffer(a_buf, a_raw);
        self.release_temp_buffer(b_buf, b_raw);
        mat
    }

    pub fn run_mul_backward(&self, a: &Mat<f32>, b: &Mat<f32>, grad_out: &Mat<f32>) -> (Mat<f32>, Mat<f32>) {
        let total = a.nrows() * a.ncols();
        let (a_buf, a_raw) = self.upload_to_temp_buffer(&Self::mat_to_flat(a));
        let (b_buf, b_raw) = self.upload_to_temp_buffer(&Self::mat_to_flat(b));
        let (go_buf, go_raw) = self.upload_to_temp_buffer(&Self::mat_to_flat(grad_out));
        let (ga_buf, ga_raw) = self.acquire_temp_buffer(total);
        let (gb_buf, gb_raw) = self.acquire_temp_buffer(total);

        self.run_compute_shader(
            self.pipeline_cache.mul_bwd.clone(),
            &[
                (0, a_buf.clone()),
                (1, b_buf.clone()),
                (2, go_buf.clone()),
                (3, ga_buf.clone()),
                (4, gb_buf.clone()),
            ],
            &[total as u32],
            total,
        );
        let ga = self.read_temp_buffer_to_mat(ga_buf, ga_raw, a.nrows(), a.ncols());
        let gb = self.read_temp_buffer_to_mat(gb_buf, gb_raw, a.nrows(), a.ncols());
        self.release_temp_buffer(a_buf, a_raw);
        self.release_temp_buffer(b_buf, b_raw);
        self.release_temp_buffer(go_buf, go_raw);
        (ga, gb)
    }

    /// Буферизованные версии.
    pub fn run_mul_forward_buffered(&self, a: &MatrixBuffer, b: &MatrixBuffer) -> MatrixBuffer {
        assert!(a.is_gpu() && b.is_gpu(), "Buffers must be GPU");
        let rows = a.rows();
        let cols = a.cols();
        let total = rows * cols;
        assert_eq!(b.rows(), rows);
        assert_eq!(b.cols(), cols);

        let a_buf = a.as_gpu_buffer().expect("GPU buffer");
        let b_buf = b.as_gpu_buffer().expect("GPU buffer");
        let out = self.allocate_gpu_matrix(rows, cols);
        let out_buf = out.as_gpu_buffer().expect("GPU buffer");

        self.run_compute_shader(
            self.pipeline_cache.mul_fwd.clone(),
            &[(0, a_buf.clone()), (1, b_buf.clone()), (2, out_buf.clone())],
            &[total as u32],
            total,
        );
        out
    }

    pub fn run_mul_backward_buffered(&self, a: &MatrixBuffer, b: &MatrixBuffer, grad_out: &MatrixBuffer) -> (MatrixBuffer, MatrixBuffer) {
        assert!(a.is_gpu() && b.is_gpu() && grad_out.is_gpu(), "Buffers must be GPU");
        let rows = a.rows();
        let cols = a.cols();
        let total = rows * cols;
        assert_eq!(b.rows(), rows);
        assert_eq!(b.cols(), cols);
        assert_eq!(grad_out.rows(), rows);
        assert_eq!(grad_out.cols(), cols);

        let a_buf = a.as_gpu_buffer().expect("GPU buffer");
        let b_buf = b.as_gpu_buffer().expect("GPU buffer");
        let go_buf = grad_out.as_gpu_buffer().expect("GPU buffer");
        let ga = self.allocate_gpu_matrix(rows, cols);
        let gb = self.allocate_gpu_matrix(rows, cols);
        let ga_buf = ga.as_gpu_buffer().expect("GPU buffer");
        let gb_buf = gb.as_gpu_buffer().expect("GPU buffer");

        self.run_compute_shader(
            self.pipeline_cache.mul_bwd.clone(),
            &[
                (0, a_buf.clone()),
                (1, b_buf.clone()),
                (2, go_buf.clone()),
                (3, ga_buf.clone()),
                (4, gb_buf.clone()),
            ],
            &[total as u32],
            total,
        );
        (ga, gb)
    }

    // ===================================================================
    // AddScalar
    // ===================================================================

    /// Старая версия.
    pub fn run_addscalar_forward(&self, input: &Mat<f32>, scalar: f32) -> Mat<f32> {
        let total = input.nrows() * input.ncols();
        let (in_buf, in_raw) = self.upload_to_temp_buffer(&Self::mat_to_flat(input));
        let (out_buf, out_raw) = self.acquire_temp_buffer(total);

        self.run_compute_shader(
            self.pipeline_cache.addscalar_fwd.clone(),
            &[(0, in_buf.clone()), (1, out_buf.clone())],
            &[total as u32, scalar.to_bits()],
            total,
        );
        let mat = self.read_temp_buffer_to_mat(out_buf, out_raw, input.nrows(), input.ncols());
        self.release_temp_buffer(in_buf, in_raw);
        mat
    }

    pub fn run_addscalar_backward(&self, grad_out: &Mat<f32>) -> Mat<f32> {
        let total = grad_out.nrows() * grad_out.ncols();
        let (go_buf, go_raw) = self.upload_to_temp_buffer(&Self::mat_to_flat(grad_out));
        let (gi_buf, gi_raw) = self.acquire_temp_buffer(total);

        self.run_compute_shader(
            self.pipeline_cache.addscalar_bwd.clone(),
            &[(0, go_buf.clone()), (1, gi_buf.clone())],
            &[total as u32],
            total,
        );
        let mat = self.read_temp_buffer_to_mat(gi_buf, gi_raw, grad_out.nrows(), grad_out.ncols());
        self.release_temp_buffer(go_buf, go_raw);
        mat
    }

    /// Буферизованные версии.
    pub fn run_addscalar_forward_buffered(&self, input: &MatrixBuffer, scalar: f32) -> MatrixBuffer {
        assert!(input.is_gpu(), "Buffer must be GPU");
        let rows = input.rows();
        let cols = input.cols();
        let total = rows * cols;

        let in_buf = input.as_gpu_buffer().expect("GPU buffer");
        let out = self.allocate_gpu_matrix(rows, cols);
        let out_buf = out.as_gpu_buffer().expect("GPU buffer");

        self.run_compute_shader(
            self.pipeline_cache.addscalar_fwd.clone(),
            &[(0, in_buf.clone()), (1, out_buf.clone())],
            &[total as u32, scalar.to_bits()],
            total,
        );
        out
    }

    pub fn run_addscalar_backward_buffered(&self, grad_out: &MatrixBuffer) -> MatrixBuffer {
        assert!(grad_out.is_gpu(), "Buffer must be GPU");
        let rows = grad_out.rows();
        let cols = grad_out.cols();
        let total = rows * cols;

        let go_buf = grad_out.as_gpu_buffer().expect("GPU buffer");
        let gi = self.allocate_gpu_matrix(rows, cols);
        let gi_buf = gi.as_gpu_buffer().expect("GPU buffer");

        self.run_compute_shader(
            self.pipeline_cache.addscalar_bwd.clone(),
            &[(0, go_buf.clone()), (1, gi_buf.clone())],
            &[total as u32],
            total,
        );
        gi
    }

    // ===================================================================
    // CrossEntropy
    // ===================================================================

    /// Старая версия (исправлен диспатч).
    pub fn run_cross_entropy_forward(&self, logits_and_target: &Mat<f32>, num_classes: usize) -> Mat<f32> {
        let batch = logits_and_target.nrows();
        let (in_buf, in_raw) = self.upload_to_temp_buffer(&Self::mat_to_flat(logits_and_target));
        let (out_buf, out_raw) = self.acquire_temp_buffer(batch);

        self.run_compute_shader_with_dispatch(
            self.pipeline_cache.cross_entropy_fwd.clone(),
            &[(0, in_buf.clone()), (1, out_buf.clone())],
            &[batch as u32, num_classes as u32],
            [batch as u32, 1, 1],
        );
        let mat = self.read_temp_buffer_to_mat(out_buf, out_raw, batch, 1);
        self.release_temp_buffer(in_buf, in_raw);
        mat
    }

    pub fn run_cross_entropy_backward(
        &self,
        logits_and_target: &Mat<f32>,
        grad_out: &Mat<f32>,
        num_classes: usize,
    ) -> Mat<f32> {
        let batch = logits_and_target.nrows();
        let total_elements = batch * (num_classes + 1);
        let (in_buf, in_raw) = self.upload_to_temp_buffer(&Self::mat_to_flat(logits_and_target));
        let (go_buf, go_raw) = self.upload_to_temp_buffer(&Self::mat_to_flat(grad_out));
        let (gi_buf, gi_raw) = self.acquire_temp_buffer(total_elements);

        self.run_compute_shader_with_dispatch(
            self.pipeline_cache.cross_entropy_bwd.clone(),
            &[(0, in_buf.clone()), (1, go_buf.clone()), (2, gi_buf.clone())],
            &[batch as u32, num_classes as u32],
            [batch as u32, 1, 1],
        );
        let mat = self.read_temp_buffer_to_mat(gi_buf, gi_raw, batch, num_classes + 1);
        self.release_temp_buffer(in_buf, in_raw);
        self.release_temp_buffer(go_buf, go_raw);
        mat
    }

    /// Буферизованные версии.
    pub fn run_cross_entropy_forward_buffered(&self, logits_and_target: &MatrixBuffer, num_classes: usize) -> MatrixBuffer {
        assert!(logits_and_target.is_gpu(), "Buffer must be GPU");
        let batch = logits_and_target.rows();
        let cols = logits_and_target.cols();
        assert_eq!(cols, num_classes + 1);

        let in_buf = logits_and_target.as_gpu_buffer().expect("GPU buffer");
        let out = self.allocate_gpu_matrix(batch, 1);
        let out_buf = out.as_gpu_buffer().expect("GPU buffer");

        self.run_compute_shader_with_dispatch(
            self.pipeline_cache.cross_entropy_fwd.clone(),
            &[(0, in_buf.clone()), (1, out_buf.clone())],
            &[batch as u32, num_classes as u32],
            [batch as u32, 1, 1],
        );
        out
    }

    pub fn run_cross_entropy_backward_buffered(
        &self,
        logits_and_target: &MatrixBuffer,
        grad_out: &MatrixBuffer,
        num_classes: usize,
    ) -> MatrixBuffer {
        assert!(logits_and_target.is_gpu() && grad_out.is_gpu(), "Buffers must be GPU");
        let batch = logits_and_target.rows();
        let cols = logits_and_target.cols();
        assert_eq!(cols, num_classes + 1);
        assert_eq!(grad_out.rows(), batch);
        assert_eq!(grad_out.cols(), 1);

        let in_buf = logits_and_target.as_gpu_buffer().expect("GPU buffer");
        let go_buf = grad_out.as_gpu_buffer().expect("GPU buffer");
        let gi = self.allocate_gpu_matrix(batch, cols);
        let gi_buf = gi.as_gpu_buffer().expect("GPU buffer");

        self.run_compute_shader_with_dispatch(
            self.pipeline_cache.cross_entropy_bwd.clone(),
            &[(0, in_buf.clone()), (1, go_buf.clone()), (2, gi_buf.clone())],
            &[batch as u32, num_classes as u32],
            [batch as u32, 1, 1],
        );
        gi
    }

    // ===================================================================
    // SumColumns (новый функционал для Этапа 4)
    // ===================================================================

    /// Буферизованная версия прямого прохода SumColumns на GPU.
    /// Суммирует все столбцы входной матрицы, создавая выход размера `(rows, 1)`.
    pub fn run_sum_columns_forward_buffered(&self, input: &MatrixBuffer) -> MatrixBuffer {
        assert!(input.is_gpu(), "Input buffer must be GPU");
        let rows = input.rows();
        let cols = input.cols();

        let in_buf = input.as_gpu_buffer().expect("GPU buffer");
        let out = self.allocate_gpu_matrix(rows, 1);
        let out_buf = out.as_gpu_buffer().expect("GPU buffer");

        let pipeline = self.pipeline_cache.reduce_pipeline();
        let push = [rows as u32];

        // Диспатч: каждая workgroup обрабатывает один столбец.
        // В шейдере reduce.comp используются gl_WorkGroupID.x как индекс столбца.
        // Количество workgroups = cols.
        self.run_compute_shader_with_dispatch(
            pipeline,
            &[(0, in_buf.clone()), (1, out_buf.clone())],
            &push,
            [cols as u32, 1, 1],
        );

        out
    }

    /// Буферизованная версия обратного прохода SumColumns на GPU.
    /// Принимает градиент размера `(rows, 1)`, возвращает градиент
    /// размера `(rows, original_cols)`, где каждый столбец повторяет входной градиент.
    ///
    /// Реализация использует временный CPU-вектор для подготовки данных,
    /// после чего данные загружаются обратно на GPU. Это допустимо на данном этапе.
    pub fn run_sum_columns_backward_buffered(
        &self,
        grad_out: &MatrixBuffer,
        original_cols: usize,
    ) -> MatrixBuffer {
        assert!(grad_out.is_gpu(), "Gradient buffer must be GPU");
        let rows = grad_out.rows();
        let grad_vec = self.download_gpu_matrix_to_vec(grad_out);

        let mut broadcast_vec = Vec::with_capacity(rows * original_cols);
        for _ in 0..original_cols {
            broadcast_vec.extend_from_slice(&grad_vec);
        }

        self.upload_vec_to_gpu_buffer(&broadcast_vec, rows, original_cols)
    }
}