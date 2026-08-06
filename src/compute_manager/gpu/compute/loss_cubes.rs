// src/compute_manager/gpu/compute/loss_cubes.rs

use faer::Mat;
use vulkano::buffer::Subbuffer;
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::{Pipeline, PipelineBindPoint};
use super::base::GpuCompute;

impl GpuCompute {
    // --- Sub ---
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

    // --- Square ---
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

    // --- Abs ---
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

    // --- Log1p ---
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

    // --- AbsDiff ---
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

    // --- Log ---
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

    // --- Neg ---
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

    // --- Mul ---
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

    // --- AddScalar ---
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

    // --- CrossEntropy (исправлен диспатч) ---
    pub fn run_cross_entropy_forward(&self, logits_and_target: &Mat<f32>, num_classes: usize) -> Mat<f32> {
        let batch = logits_and_target.nrows();
        let (in_buf, in_raw) = self.upload_to_temp_buffer(&Self::mat_to_flat(logits_and_target));
        let (out_buf, out_raw) = self.acquire_temp_buffer(batch);

        self.run_compute_shader_with_dispatch(
            self.pipeline_cache.cross_entropy_fwd.clone(),
            &[(0, in_buf.clone()), (1, out_buf.clone())],
            &[batch as u32, num_classes as u32],
            [batch as u32, 1, 1],      // <-- правильный диспатч
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
            [batch as u32, 1, 1],      // <-- правильный диспатч
        );
        let mat = self.read_temp_buffer_to_mat(gi_buf, gi_raw, batch, num_classes + 1);
        self.release_temp_buffer(in_buf, in_raw);
        self.release_temp_buffer(go_buf, go_raw);
        mat
    }
}