pub mod pipeline;   // <-- новый модуль

use crate::compute_manager::gpu::compute::GpuCompute;
use crate::compute_manager::matrix_buffer::MatrixBufferHandle;

impl GpuCompute {
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

        // Используем новый пайплайн из собственной структуры DualAnchor
        let pipeline = self.dual_anchor_pipelines().forward.clone();
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
        grad_min_cpu: &MatrixBufferHandle,
        grad_max_cpu: &MatrixBufferHandle,
        grad_alpha_cpu: &MatrixBufferHandle,
    ) {
        assert!(input.is_gpu() && grad_out.is_gpu(), "Handles must be GPU");
        assert!(grad_input.is_gpu() && grad_min.is_gpu() && grad_max.is_gpu() && grad_alpha.is_gpu(),
            "Gradient handles must be GPU");
        assert!(!grad_min_cpu.is_gpu() && !grad_max_cpu.is_gpu() && !grad_alpha_cpu.is_gpu(),
            "CPU gradient handles must not be GPU");
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
        assert_eq!(grad_min_cpu.rows(), 1);
        assert_eq!(grad_min_cpu.cols(), features);
        assert_eq!(grad_max_cpu.rows(), 1);
        assert_eq!(grad_max_cpu.cols(), features);
        assert_eq!(grad_alpha_cpu.rows(), 1);
        assert_eq!(grad_alpha_cpu.cols(), 1);

        let in_buf = self.get_gpu_subbuffer_from_handle(input);
        let go_buf = self.get_gpu_subbuffer_from_handle(grad_out);
        let (min_buf, min_raw) = self.upload_to_temp_buffer(min_vals);
        let (max_buf, max_raw) = self.upload_to_temp_buffer(max_vals);
        let gi_buf = self.get_gpu_subbuffer_from_handle(grad_input);
        let gmin_buf = self.get_gpu_subbuffer_from_handle(grad_min);
        let gmax_buf = self.get_gpu_subbuffer_from_handle(grad_max);
        let galpha_buf = self.get_gpu_subbuffer_from_handle(grad_alpha);

        // Используем новый пайплайн из собственной структуры DualAnchor
        let pipeline = self.dual_anchor_pipelines().backward.clone();
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

        self.copy_gpu_to_cpu_handle(grad_min, grad_min_cpu);
        self.copy_gpu_to_cpu_handle(grad_max, grad_max_cpu);
        self.copy_gpu_to_cpu_handle(grad_alpha, grad_alpha_cpu);
    }
}