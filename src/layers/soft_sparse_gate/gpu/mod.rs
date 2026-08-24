pub mod pipeline;   // <-- новый модуль

use crate::compute_manager::gpu::compute::GpuCompute;
use crate::compute_manager::matrix_buffer::MatrixBufferHandle;

impl GpuCompute {
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

        // Используем новый пайплайн из собственной структуры SoftSparseGate
        let pipeline = self.soft_sparse_gate_pipelines().forward.clone();
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
        grad_thresh_cpu: &MatrixBufferHandle,
    ) {
        assert!(input.is_gpu() && grad_out.is_gpu(), "Handles must be GPU");
        assert!(grad_input.is_gpu() && grad_thresh.is_gpu(), "Grad handles must be GPU");
        assert!(!grad_thresh_cpu.is_gpu(), "grad_thresh_cpu must be CPU");
        let batch = input.rows();
        let features = input.cols();
        let total = batch * features;
        assert_eq!(grad_out.rows(), batch);
        assert_eq!(grad_out.cols(), features);
        assert_eq!(grad_input.rows(), batch);
        assert_eq!(grad_input.cols(), features);
        assert_eq!(grad_thresh.rows(), 1);
        assert_eq!(grad_thresh.cols(), features);
        assert_eq!(grad_thresh_cpu.rows(), 1);
        assert_eq!(grad_thresh_cpu.cols(), features);

        let in_buf = self.get_gpu_subbuffer_from_handle(input);
        let go_buf = self.get_gpu_subbuffer_from_handle(grad_out);
        let (thresh_buf, th_raw) = self.upload_to_temp_buffer(thresholds);
        let gi_buf = self.get_gpu_subbuffer_from_handle(grad_input);
        let gthresh_buf = self.get_gpu_subbuffer_from_handle(grad_thresh);

        // Используем новый пайплайн из собственной структуры SoftSparseGate
        let pipeline = self.soft_sparse_gate_pipelines().backward.clone();
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

        // Копируем градиент порогов из GPU в CPU-буфер
        self.copy_gpu_to_cpu_handle(grad_thresh, grad_thresh_cpu);
    }
}