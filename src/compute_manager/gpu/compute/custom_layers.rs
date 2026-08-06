// src/compute_manager/gpu/compute/custom_layers.rs

use faer::Mat;
use vulkano::buffer::Subbuffer;
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::{Pipeline, PipelineBindPoint};
use super::base::GpuCompute;

impl GpuCompute {
    // ---------- Memory ----------
    pub fn init_memory_state(&mut self, features: usize, _alpha: f32) {
        let mut state = Vec::with_capacity(2 * features);
        state.resize(features, f32::MAX);
        state.resize(2 * features, f32::MIN);
        let (buf, raw_id) = self.upload_to_temp_buffer(&state);
        self.memory_state = Some(buf);
        self.memory_state_id = Some(raw_id);
    }

    pub fn run_memory_forward(
        &self,
        input: &Mat<f32>,
        alpha: f32,
        state: &Subbuffer<[f32]>,
    ) -> Mat<f32> {
        let batch = input.nrows();
        let features = input.ncols();
        let total = batch * features;
        let (in_buf, in_raw) = self.upload_to_temp_buffer(&Self::mat_to_flat(input));
        let (out_buf, out_raw) = self.acquire_temp_buffer(total);

        let pipeline = self.pipeline_cache.memory_fwd.clone();
        let push = [batch as u32, features as u32, alpha.to_bits()];
        self.run_compute_shader(
            pipeline,
            &[(0, in_buf.clone()), (1, state.clone()), (2, out_buf.clone())],
            &push,
            total,
        );

        let mat = self.read_temp_buffer_to_mat(out_buf, out_raw, batch, features);
        self.release_temp_buffer(in_buf, in_raw);
        mat
    }

    pub fn run_memory_backward(
        &self,
        grad_out: &Mat<f32>,
        alpha: f32,
    ) -> Mat<f32> {
        let total = grad_out.nrows() * grad_out.ncols();
        let (go_buf, go_raw) = self.upload_to_temp_buffer(&Self::mat_to_flat(grad_out));
        let (gi_buf, gi_raw) = self.acquire_temp_buffer(total);

        let pipeline = self.pipeline_cache.memory_bwd.clone();
        let push = [total as u32, alpha.to_bits()];
        self.run_compute_shader(
            pipeline,
            &[(0, go_buf.clone()), (1, gi_buf.clone())],
            &push,
            total,
        );

        let mat = self.read_temp_buffer_to_mat(gi_buf, gi_raw, grad_out.nrows(), grad_out.ncols());
        self.release_temp_buffer(go_buf, go_raw);
        mat
    }

    // ---------- SoftSparseGate ----------
    pub fn run_softsparse_forward(
        &self,
        input: &Mat<f32>,
        thresholds: &[f32],
        temperature: f32,
    ) -> Mat<f32> {
        let batch = input.nrows();
        let features = input.ncols();
        let total = batch * features;
        let (in_buf, in_raw) = self.upload_to_temp_buffer(&Self::mat_to_flat(input));
        let (thresh_buf, th_raw) = self.upload_to_temp_buffer(thresholds);
        let (out_buf, out_raw) = self.acquire_temp_buffer(total);

        let pipeline = self.pipeline_cache.softsparse_fwd.clone();
        let push = [total as u32, temperature.to_bits(), features as u32];
        self.run_compute_shader(
            pipeline,
            &[(0, in_buf.clone()), (1, thresh_buf.clone()), (2, out_buf.clone())],
            &push,
            total,
        );

        let mat = self.read_temp_buffer_to_mat(out_buf, out_raw, batch, features);
        self.release_temp_buffer(in_buf, in_raw);
        self.release_temp_buffer(thresh_buf, th_raw);
        mat
    }

    pub fn run_softsparse_backward(
        &self,
        input: &Mat<f32>,
        grad_out: &Mat<f32>,
        thresholds: &[f32],
        temperature: f32,
    ) -> (Mat<f32>, Vec<f32>) {
        let batch = input.nrows();
        let features = input.ncols();
        let total = batch * features;
        let (in_buf, in_raw) = self.upload_to_temp_buffer(&Self::mat_to_flat(input));
        let (go_buf, go_raw) = self.upload_to_temp_buffer(&Self::mat_to_flat(grad_out));
        let (thresh_buf, th_raw) = self.upload_to_temp_buffer(thresholds);
        let (gi_buf, gi_raw) = self.acquire_temp_buffer(total);
        let (gthresh_buf, gth_raw) = self.acquire_temp_buffer(features);

        let pipeline = self.pipeline_cache.softsparse_bwd.clone();
        let push = [total as u32, temperature.to_bits(), features as u32];
        self.run_compute_shader(
            pipeline,
            &[
                (0, in_buf.clone()),
                (1, go_buf.clone()),
                (2, thresh_buf.clone()),
                (3, gi_buf.clone()),
                (4, gthresh_buf.clone()),
            ],
            &push,
            total,
        );

        let gi = self.read_temp_buffer_to_mat(gi_buf, gi_raw, batch, features);
        let gthresh = self.read_temp_buffer_to_mat(gthresh_buf, gth_raw, 1, features);
        let grad_thresh: Vec<f32> = (0..features).map(|c| gthresh[(0, c)]).collect();

        self.release_temp_buffer(in_buf, in_raw);
        self.release_temp_buffer(go_buf, go_raw);
        self.release_temp_buffer(thresh_buf, th_raw);
        (gi, grad_thresh)
    }

    // ---------- SoftKeepGate ----------
    pub fn run_softkeep_forward(
        &self,
        input: &Mat<f32>,
        thresholds: &[f32],
        temperature: f32,
    ) -> Mat<f32> {
        let batch = input.nrows();
        let features = input.ncols();
        let total = batch * features;
        let (in_buf, in_raw) = self.upload_to_temp_buffer(&Self::mat_to_flat(input));
        let (thresh_buf, th_raw) = self.upload_to_temp_buffer(thresholds);
        let (out_buf, out_raw) = self.acquire_temp_buffer(total);

        let pipeline = self.pipeline_cache.softkeep_fwd.clone();
        let push = [total as u32, temperature.to_bits(), features as u32];
        self.run_compute_shader(
            pipeline,
            &[(0, in_buf.clone()), (1, thresh_buf.clone()), (2, out_buf.clone())],
            &push,
            total,
        );

        let mat = self.read_temp_buffer_to_mat(out_buf, out_raw, batch, features);
        self.release_temp_buffer(in_buf, in_raw);
        self.release_temp_buffer(thresh_buf, th_raw);
        mat
    }

    pub fn run_softkeep_backward(
        &self,
        input: &Mat<f32>,
        grad_out: &Mat<f32>,
        thresholds: &[f32],
        temperature: f32,
    ) -> (Mat<f32>, Vec<f32>) {
        let batch = input.nrows();
        let features = input.ncols();
        let total = batch * features;
        let (in_buf, in_raw) = self.upload_to_temp_buffer(&Self::mat_to_flat(input));
        let (go_buf, go_raw) = self.upload_to_temp_buffer(&Self::mat_to_flat(grad_out));
        let (thresh_buf, th_raw) = self.upload_to_temp_buffer(thresholds);
        let (gi_buf, gi_raw) = self.acquire_temp_buffer(total);
        let (gthresh_buf, gth_raw) = self.acquire_temp_buffer(features);

        let pipeline = self.pipeline_cache.softkeep_bwd.clone();
        let push = [total as u32, temperature.to_bits(), features as u32];
        self.run_compute_shader(
            pipeline,
            &[
                (0, in_buf.clone()),
                (1, go_buf.clone()),
                (2, thresh_buf.clone()),
                (3, gi_buf.clone()),
                (4, gthresh_buf.clone()),
            ],
            &push,
            total,
        );

        let gi = self.read_temp_buffer_to_mat(gi_buf, gi_raw, batch, features);
        let gthresh = self.read_temp_buffer_to_mat(gthresh_buf, gth_raw, 1, features);
        let grad_thresh: Vec<f32> = (0..features).map(|c| gthresh[(0, c)]).collect();

        self.release_temp_buffer(in_buf, in_raw);
        self.release_temp_buffer(go_buf, go_raw);
        self.release_temp_buffer(thresh_buf, th_raw);
        (gi, grad_thresh)
    }

    // ---------- DualAnchor ----------
    pub fn run_dualanchor_forward(
        &self,
        input: &Mat<f32>,
        min_vals: &[f32],
        max_vals: &[f32],
        alpha: f32,
    ) -> Mat<f32> {
        let batch = input.nrows();
        let features = input.ncols();
        let total = batch * features;
        let (in_buf, in_raw) = self.upload_to_temp_buffer(&Self::mat_to_flat(input));
        let (min_buf, min_raw) = self.upload_to_temp_buffer(min_vals);
        let (max_buf, max_raw) = self.upload_to_temp_buffer(max_vals);
        let (out_buf, out_raw) = self.acquire_temp_buffer(total);

        let pipeline = self.pipeline_cache.dualanchor_fwd.clone();
        let push = [total as u32, features as u32, alpha.to_bits()];
        self.run_compute_shader(
            pipeline,
            &[
                (0, in_buf.clone()),
                (1, min_buf.clone()),
                (2, max_buf.clone()),
                (3, out_buf.clone()),
            ],
            &push,
            total,
        );

        let mat = self.read_temp_buffer_to_mat(out_buf, out_raw, batch, features);
        self.release_temp_buffer(in_buf, in_raw);
        self.release_temp_buffer(min_buf, min_raw);
        self.release_temp_buffer(max_buf, max_raw);
        mat
    }

    pub fn run_dualanchor_backward(
        &self,
        input: &Mat<f32>,
        grad_out: &Mat<f32>,
        min_vals: &[f32],
        max_vals: &[f32],
        alpha: f32,
    ) -> (Mat<f32>, Vec<f32>) {
        let batch = input.nrows();
        let features = input.ncols();
        let total = batch * features;
        let (in_buf, in_raw) = self.upload_to_temp_buffer(&Self::mat_to_flat(input));
        let (go_buf, go_raw) = self.upload_to_temp_buffer(&Self::mat_to_flat(grad_out));
        let (min_buf, min_raw) = self.upload_to_temp_buffer(min_vals);
        let (max_buf, max_raw) = self.upload_to_temp_buffer(max_vals);
        let (gi_buf, gi_raw) = self.acquire_temp_buffer(total);
        let (gmin_buf, gmin_raw) = self.acquire_temp_buffer(features);
        let (gmax_buf, gmax_raw) = self.acquire_temp_buffer(features);
        let (galpha_buf, galpha_raw) = self.acquire_temp_buffer(1);

        let pipeline = self.pipeline_cache.dualanchor_bwd.clone();
        let push = [total as u32, features as u32, alpha.to_bits()];
        self.run_compute_shader(
            pipeline,
            &[
                (0, in_buf.clone()),
                (1, go_buf.clone()),
                (2, min_buf.clone()),
                (3, max_buf.clone()),
                (4, gi_buf.clone()),
                (5, gmin_buf.clone()),
                (6, gmax_buf.clone()),
                (7, galpha_buf.clone()),
            ],
            &push,
            total,
        );

        let gi = self.read_temp_buffer_to_mat(gi_buf, gi_raw, batch, features);
        let gmin_mat = self.read_temp_buffer_to_mat(gmin_buf, gmin_raw, 1, features);
        let gmax_mat = self.read_temp_buffer_to_mat(gmax_buf, gmax_raw, 1, features);
        let galpha_mat = self.read_temp_buffer_to_mat(galpha_buf, galpha_raw, 1, 1);

        let mut grad = Vec::with_capacity(2 * features + 1);
        for c in 0..features {
            grad.push(gmin_mat[(0, c)]);
        }
        for c in 0..features {
            grad.push(gmax_mat[(0, c)]);
        }
        grad.push(galpha_mat[(0, 0)]);

        self.release_temp_buffer(in_buf, in_raw);
        self.release_temp_buffer(go_buf, go_raw);
        self.release_temp_buffer(min_buf, min_raw);
        self.release_temp_buffer(max_buf, max_raw);
        (gi, grad)
    }
}