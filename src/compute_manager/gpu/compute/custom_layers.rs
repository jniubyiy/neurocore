// src/compute_manager/gpu/compute/custom_layers.rs

use super::base::GpuCompute;
use crate::compute_manager::matrix_buffer::MatrixBufferHandle;

impl GpuCompute {
    // ===================================================================
    // Memory
    // ===================================================================

    /// Инициализирует внутреннее состояние слоя Memory.
    /// Вызывается один раз перед первым использованием.
    pub fn init_memory_state(&mut self, features: usize, _alpha: f32) {
        let mut state = Vec::with_capacity(2 * features);
        state.resize(features, f32::MAX);
        state.resize(2 * features, f32::MIN);
        let (buf, raw_id) = self.upload_to_temp_buffer(&state);
        self.memory_state = Some(buf);
        self.memory_state_id = Some(raw_id);
    }

    pub fn run_memory_forward_buffered_handle(
        &self,
        input: &MatrixBufferHandle,
        output: &MatrixBufferHandle,
        alpha: f32,
    ) {
        assert!(input.is_gpu() && output.is_gpu(), "Handles must be GPU");
        let batch = input.rows();
        let features = input.cols();
        let total = batch * features;
        assert_eq!(output.rows(), batch);
        assert_eq!(output.cols(), features);

        let in_buf = self.get_gpu_subbuffer_from_handle(input);
        let state = self.memory_state.as_ref().expect("Memory state not initialized");
        let out_buf = self.get_gpu_subbuffer_from_handle(output);

        let pipeline = self.pipeline_cache.memory_fwd.clone();
        let push = [batch as u32, features as u32, alpha.to_bits()];
        self.run_compute_shader(
            pipeline,
            &[(0, in_buf), (1, state.clone()), (2, out_buf)],
            &push,
            total,
        );
    }

    pub fn run_memory_backward_buffered_handle(
        &self,
        grad_out: &MatrixBufferHandle,
        grad_input: &MatrixBufferHandle,
        alpha: f32,
    ) {
        assert!(grad_out.is_gpu() && grad_input.is_gpu(), "Handles must be GPU");
        let rows = grad_out.rows();
        let cols = grad_out.cols();
        let total = rows * cols;
        assert_eq!(grad_input.rows(), rows);
        assert_eq!(grad_input.cols(), cols);

        let go_buf = self.get_gpu_subbuffer_from_handle(grad_out);
        let gi_buf = self.get_gpu_subbuffer_from_handle(grad_input);

        let pipeline = self.pipeline_cache.memory_bwd.clone();
        let push = [total as u32, alpha.to_bits()];
        self.run_compute_shader(
            pipeline,
            &[(0, go_buf), (1, gi_buf)],
            &push,
            total,
        );
    }

    // ===================================================================
    // SoftSparseGate
    // ===================================================================

    pub fn run_softsparse_forward_buffered_handle(
        &self,
        input: &MatrixBufferHandle,
        thresholds: &[f32],
        temperature: f32,
        output: &MatrixBufferHandle,
    ) {
        assert!(input.is_gpu() && output.is_gpu(), "Handles must be GPU");
        let batch = input.rows();
        let features = input.cols();
        let total = batch * features;
        assert_eq!(output.rows(), batch);
        assert_eq!(output.cols(), features);

        let in_buf = self.get_gpu_subbuffer_from_handle(input);
        let (thresh_buf, th_raw) = self.upload_to_temp_buffer(thresholds);
        let out_buf = self.get_gpu_subbuffer_from_handle(output);

        let pipeline = self.pipeline_cache.softsparse_fwd.clone();
        let push = [total as u32, temperature.to_bits(), features as u32];
        self.run_compute_shader(
            pipeline,
            &[(0, in_buf), (1, thresh_buf.clone()), (2, out_buf)],
            &push,
            total,
        );

        self.release_temp_buffer(thresh_buf, th_raw);
    }

    pub fn run_softsparse_backward_buffered_handle(
        &self,
        input: &MatrixBufferHandle,
        grad_out: &MatrixBufferHandle,
        thresholds: &[f32],
        temperature: f32,
        grad_input: &MatrixBufferHandle,
        grad_thresh: &MatrixBufferHandle,
    ) -> Vec<f32> {
        assert!(input.is_gpu() && grad_out.is_gpu(), "Handles must be GPU");
        assert!(grad_input.is_gpu() && grad_thresh.is_gpu(), "Grad handles must be GPU");
        let batch = input.rows();
        let features = input.cols();
        let total = batch * features;
        assert_eq!(grad_out.rows(), batch);
        assert_eq!(grad_out.cols(), features);
        assert_eq!(grad_input.rows(), batch);
        assert_eq!(grad_input.cols(), features);
        assert_eq!(grad_thresh.rows(), 1);
        assert_eq!(grad_thresh.cols(), features);

        let in_buf = self.get_gpu_subbuffer_from_handle(input);
        let go_buf = self.get_gpu_subbuffer_from_handle(grad_out);
        let (thresh_buf, th_raw) = self.upload_to_temp_buffer(thresholds);
        let gi_buf = self.get_gpu_subbuffer_from_handle(grad_input);
        let gthresh_buf = self.get_gpu_subbuffer_from_handle(grad_thresh);

        let pipeline = self.pipeline_cache.softsparse_bwd.clone();
        let push = [total as u32, temperature.to_bits(), features as u32];
        self.run_compute_shader(
            pipeline,
            &[
                (0, in_buf),
                (1, go_buf),
                (2, thresh_buf.clone()),
                (3, gi_buf),
                (4, gthresh_buf),
            ],
            &push,
            total,
        );

        self.release_temp_buffer(thresh_buf, th_raw);

        let gthresh_vec = self.download_gpu_handle_to_vec(grad_thresh);
        gthresh_vec
    }

    // ===================================================================
    // SoftKeepGate
    // ===================================================================

    pub fn run_softkeep_forward_buffered_handle(
        &self,
        input: &MatrixBufferHandle,
        thresholds: &[f32],
        temperature: f32,
        output: &MatrixBufferHandle,
    ) {
        assert!(input.is_gpu() && output.is_gpu(), "Handles must be GPU");
        let batch = input.rows();
        let features = input.cols();
        let total = batch * features;
        assert_eq!(output.rows(), batch);
        assert_eq!(output.cols(), features);

        let in_buf = self.get_gpu_subbuffer_from_handle(input);
        let (thresh_buf, th_raw) = self.upload_to_temp_buffer(thresholds);
        let out_buf = self.get_gpu_subbuffer_from_handle(output);

        let pipeline = self.pipeline_cache.softkeep_fwd.clone();
        let push = [total as u32, temperature.to_bits(), features as u32];
        self.run_compute_shader(
            pipeline,
            &[(0, in_buf), (1, thresh_buf.clone()), (2, out_buf)],
            &push,
            total,
        );

        self.release_temp_buffer(thresh_buf, th_raw);
    }

    pub fn run_softkeep_backward_buffered_handle(
        &self,
        input: &MatrixBufferHandle,
        grad_out: &MatrixBufferHandle,
        thresholds: &[f32],
        temperature: f32,
        grad_input: &MatrixBufferHandle,
        grad_thresh: &MatrixBufferHandle,
    ) -> Vec<f32> {
        assert!(input.is_gpu() && grad_out.is_gpu(), "Handles must be GPU");
        assert!(grad_input.is_gpu() && grad_thresh.is_gpu(), "Grad handles must be GPU");
        let batch = input.rows();
        let features = input.cols();
        let total = batch * features;
        assert_eq!(grad_out.rows(), batch);
        assert_eq!(grad_out.cols(), features);
        assert_eq!(grad_input.rows(), batch);
        assert_eq!(grad_input.cols(), features);
        assert_eq!(grad_thresh.rows(), 1);
        assert_eq!(grad_thresh.cols(), features);

        let in_buf = self.get_gpu_subbuffer_from_handle(input);
        let go_buf = self.get_gpu_subbuffer_from_handle(grad_out);
        let (thresh_buf, th_raw) = self.upload_to_temp_buffer(thresholds);
        let gi_buf = self.get_gpu_subbuffer_from_handle(grad_input);
        let gthresh_buf = self.get_gpu_subbuffer_from_handle(grad_thresh);

        let pipeline = self.pipeline_cache.softkeep_bwd.clone();
        let push = [total as u32, temperature.to_bits(), features as u32];
        self.run_compute_shader(
            pipeline,
            &[
                (0, in_buf),
                (1, go_buf),
                (2, thresh_buf.clone()),
                (3, gi_buf),
                (4, gthresh_buf),
            ],
            &push,
            total,
        );

        self.release_temp_buffer(thresh_buf, th_raw);

        let gthresh_vec = self.download_gpu_handle_to_vec(grad_thresh);
        gthresh_vec
    }

    // ===================================================================
    // DualAnchor
    // ===================================================================

    pub fn run_dualanchor_forward_buffered_handle(
        &self,
        input: &MatrixBufferHandle,
        min_vals: &[f32],
        max_vals: &[f32],
        alpha: f32,
        output: &MatrixBufferHandle,
    ) {
        assert!(input.is_gpu() && output.is_gpu(), "Handles must be GPU");
        let batch = input.rows();
        let features = input.cols();
        let total = batch * features;
        assert_eq!(output.rows(), batch);
        assert_eq!(output.cols(), features);

        let in_buf = self.get_gpu_subbuffer_from_handle(input);
        let (min_buf, min_raw) = self.upload_to_temp_buffer(min_vals);
        let (max_buf, max_raw) = self.upload_to_temp_buffer(max_vals);
        let out_buf = self.get_gpu_subbuffer_from_handle(output);

        let pipeline = self.pipeline_cache.dualanchor_fwd.clone();
        let push = [total as u32, features as u32, alpha.to_bits()];
        self.run_compute_shader(
            pipeline,
            &[
                (0, in_buf),
                (1, min_buf.clone()),
                (2, max_buf.clone()),
                (3, out_buf),
            ],
            &push,
            total,
        );

        self.release_temp_buffer(min_buf, min_raw);
        self.release_temp_buffer(max_buf, max_raw);
    }

    pub fn run_dualanchor_backward_buffered_handle(
        &self,
        input: &MatrixBufferHandle,
        grad_out: &MatrixBufferHandle,
        min_vals: &[f32],
        max_vals: &[f32],
        alpha: f32,
        grad_input: &MatrixBufferHandle,
        grad_min: &MatrixBufferHandle,
        grad_max: &MatrixBufferHandle,
        grad_alpha: &MatrixBufferHandle,
    ) -> Vec<f32> {
        assert!(input.is_gpu() && grad_out.is_gpu(), "Handles must be GPU");
        assert!(grad_input.is_gpu() && grad_min.is_gpu() && grad_max.is_gpu() && grad_alpha.is_gpu(),
            "Gradient handles must be GPU");
        let batch = input.rows();
        let features = input.cols();
        let total = batch * features;
        assert_eq!(grad_out.rows(), batch);
        assert_eq!(grad_out.cols(), features);
        assert_eq!(grad_input.rows(), batch);
        assert_eq!(grad_input.cols(), features);
        assert_eq!(grad_min.rows(), 1);
        assert_eq!(grad_min.cols(), features);
        assert_eq!(grad_max.rows(), 1);
        assert_eq!(grad_max.cols(), features);
        assert_eq!(grad_alpha.rows(), 1);
        assert_eq!(grad_alpha.cols(), 1);

        let in_buf = self.get_gpu_subbuffer_from_handle(input);
        let go_buf = self.get_gpu_subbuffer_from_handle(grad_out);
        let (min_buf, min_raw) = self.upload_to_temp_buffer(min_vals);
        let (max_buf, max_raw) = self.upload_to_temp_buffer(max_vals);
        let gi_buf = self.get_gpu_subbuffer_from_handle(grad_input);
        let gmin_buf = self.get_gpu_subbuffer_from_handle(grad_min);
        let gmax_buf = self.get_gpu_subbuffer_from_handle(grad_max);
        let galpha_buf = self.get_gpu_subbuffer_from_handle(grad_alpha);

        let pipeline = self.pipeline_cache.dualanchor_bwd.clone();
        let push = [total as u32, features as u32, alpha.to_bits()];
        self.run_compute_shader(
            pipeline,
            &[
                (0, in_buf),
                (1, go_buf),
                (2, min_buf.clone()),
                (3, max_buf.clone()),
                (4, gi_buf),
                (5, gmin_buf),
                (6, gmax_buf),
                (7, galpha_buf),
            ],
            &push,
            total,
        );

        self.release_temp_buffer(min_buf, min_raw);
        self.release_temp_buffer(max_buf, max_raw);

        let grad_min_vec = self.download_gpu_handle_to_vec(grad_min);
        let grad_max_vec = self.download_gpu_handle_to_vec(grad_max);
        let grad_alpha_vec = self.download_gpu_handle_to_vec(grad_alpha);
        let mut combined = Vec::with_capacity(2 * features + 1);
        combined.extend_from_slice(&grad_min_vec);
        combined.extend_from_slice(&grad_max_vec);
        combined.extend_from_slice(&grad_alpha_vec);
        combined
    }
}