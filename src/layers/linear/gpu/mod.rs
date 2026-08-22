// src/layers/linear/gpu/mod.rs

use crate::compute_manager::gpu::compute::GpuCompute;
use crate::compute_manager::matrix_buffer::MatrixBufferHandle;

impl GpuCompute {
    pub fn run_linear_forward_buffered_handle(
        &self,
        input: &MatrixBufferHandle,
        weight: &MatrixBufferHandle,
        bias: &[f32],
        output: &MatrixBufferHandle,
    ) {
        assert!(input.is_gpu(), "Input handle must be GPU");
        assert!(weight.is_gpu(), "Weight handle must be GPU");
        assert!(output.is_gpu(), "Output handle must be GPU");

        let batch = input.rows();
        let in_features = input.cols();
        let out_features = weight.rows();
        assert_eq!(weight.cols(), in_features, "Weight shape mismatch");
        assert_eq!(output.rows(), batch, "Output rows mismatch");
        assert_eq!(output.cols(), out_features, "Output cols mismatch");
        assert_eq!(bias.len(), out_features, "Bias length mismatch");

        let in_buf = self.get_gpu_subbuffer_from_handle(input);
        let w_buf = self.get_gpu_subbuffer_from_handle(weight);
        let out_buf = self.get_gpu_subbuffer_from_handle(output);

        let (bias_buf, bias_raw) = self.upload_to_temp_buffer(bias);

        let pipeline = self.pipeline_cache.linear_fwd.clone();
        let push = [batch as u32, in_features as u32, out_features as u32];

        self.run_compute_shader_with_dispatch(
            pipeline,
            &[
                (0, in_buf),
                (1, w_buf),
                (2, bias_buf.clone()),
                (3, out_buf),
            ],
            &push,
            [((batch * out_features + 255) / 256) as u32, 1, 1],
        );

        self.release_temp_buffer(bias_buf, bias_raw);
    }

    pub fn run_linear_backward_buffered_handle(
        &self,
        input: &MatrixBufferHandle,
        weight: &MatrixBufferHandle,
        grad_output: &MatrixBufferHandle,
        grad_input: &MatrixBufferHandle,
        grad_weight: &MatrixBufferHandle,
        grad_bias_handle: &MatrixBufferHandle,
        grad_bias_cpu: &MatrixBufferHandle,
    ) {
        assert!(input.is_gpu(), "Input handle must be GPU");
        assert!(weight.is_gpu(), "Weight handle must be GPU");
        assert!(grad_output.is_gpu(), "grad_output handle must be GPU");
        assert!(grad_input.is_gpu(), "grad_input handle must be GPU");
        assert!(grad_weight.is_gpu(), "grad_weight handle must be GPU");
        assert!(grad_bias_handle.is_gpu(), "grad_bias_handle must be GPU");

        let batch = input.rows();
        let in_features = input.cols();
        let out_features = weight.rows();
        assert_eq!(weight.cols(), in_features, "Weight shape mismatch");
        assert_eq!(grad_input.rows(), batch, "grad_input rows mismatch");
        assert_eq!(grad_input.cols(), in_features, "grad_input cols mismatch");
        assert_eq!(grad_weight.rows(), out_features, "grad_weight rows mismatch");
        assert_eq!(grad_weight.cols(), in_features, "grad_weight cols mismatch");
        assert_eq!(grad_output.rows(), batch, "grad_output rows mismatch");
        assert_eq!(grad_output.cols(), out_features, "grad_output cols mismatch");
        assert_eq!(grad_bias_handle.rows(), 1, "grad_bias_handle rows must be 1");
        assert_eq!(grad_bias_handle.cols(), out_features, "grad_bias_handle cols mismatch");
        assert!(!grad_bias_cpu.is_gpu(), "grad_bias_cpu must be CPU");
        assert_eq!(grad_bias_cpu.rows(), 1, "grad_bias_cpu rows must be 1");
        assert_eq!(grad_bias_cpu.cols(), out_features, "grad_bias_cpu cols mismatch");

        let x_buf = self.get_gpu_subbuffer_from_handle(input);
        let w_buf = self.get_gpu_subbuffer_from_handle(weight);
        let dout_buf = self.get_gpu_subbuffer_from_handle(grad_output);
        let dx_buf = self.get_gpu_subbuffer_from_handle(grad_input);
        let dw_buf = self.get_gpu_subbuffer_from_handle(grad_weight);
        let db_buf = self.get_gpu_subbuffer_from_handle(grad_bias_handle);

        self.fill_gpu_handle(grad_weight, 0.0);
        self.fill_gpu_handle(grad_bias_handle, 0.0);

        let pipeline = self.pipeline_cache.linear_bwd.clone();
        let push = [batch as u32, in_features as u32, out_features as u32];

        self.run_compute_shader_with_dispatch(
            pipeline,
            &[
                (0, x_buf),
                (1, w_buf),
                (2, dout_buf),
                (3, dx_buf),
                (4, dw_buf),
                (5, db_buf),
            ],
            &push,
            [((batch + 255) / 256) as u32, 1, 1],
        );

        self.copy_gpu_to_cpu_handle(grad_bias_handle, grad_bias_cpu);
    }
}