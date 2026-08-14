// src/compute_manager/gpu/compute/linear.rs

use super::base::GpuCompute;
use crate::compute_manager::matrix_buffer::MatrixBuffer;
use crate::compute_manager::matrix_buffer::MatrixBufferHandle;

impl GpuCompute {
    // ===================================================================
    // Буферизованные версии для MatrixBuffer
    // ===================================================================

    pub fn run_linear_forward_buffered(
        &self,
        input: &MatrixBuffer,
        weight: &MatrixBuffer,
        bias: &[f32],
    ) -> MatrixBuffer {
        assert!(input.is_gpu() && weight.is_gpu(), "Buffers must be GPU");
        let batch = input.rows();
        let in_features = input.cols();
        let out_features = weight.rows();
        assert_eq!(weight.cols(), in_features, "Weight shape mismatch");
        assert_eq!(bias.len(), out_features, "Bias length mismatch");

        let weight_t = self.transpose_gpu_matrix(weight);
        let mut out = self.run_mat_mul_buffered(input, &weight_t);

        let mut out_vec = self.download_gpu_matrix_to_vec(&out);
        for c in 0..out_features {
            let bias_val = bias[c];
            for r in 0..batch {
                out_vec[c * batch + r] += bias_val;
            }
        }
        out = self.upload_vec_to_gpu_buffer(&out_vec, batch, out_features);
        out
    }

    pub fn run_linear_backward_buffered(
        &self,
        input: &MatrixBuffer,
        weight: &MatrixBuffer,
        grad_output: &MatrixBuffer,
    ) -> (MatrixBuffer, MatrixBuffer, Vec<f32>) {
        assert!(
            input.is_gpu() && weight.is_gpu() && grad_output.is_gpu(),
            "Buffers must be GPU"
        );
        let batch = input.rows();
        let in_features = input.cols();
        let out_features = grad_output.cols();
        assert_eq!(weight.rows(), out_features, "Weight shape mismatch");
        assert_eq!(weight.cols(), in_features, "Weight shape mismatch");
        assert_eq!(input.rows(), batch, "Batch mismatch");

        let grad_input = self.run_mat_mul_buffered(grad_output, weight);

        let grad_output_t = self.transpose_gpu_matrix(grad_output);
        let grad_weight = self.run_mat_mul_buffered(&grad_output_t, input);

        let go_vec = self.download_gpu_matrix_to_vec(grad_output);
        let grad_bias: Vec<f32> = (0..out_features)
            .map(|c| (0..batch).map(|r| go_vec[c * batch + r]).sum())
            .collect();

        (grad_input, grad_weight, grad_bias)
    }

    // ===================================================================
    // НОВЫЕ Handle-версии (MatrixBufferHandle)
    // ===================================================================

    /// Прямой проход Linear на GPU с использованием MatrixBufferHandle.
    /// `input`, `weight` и `output` должны быть GPU-буферами.
    /// Веса передаются в формате (out_features, in_features).
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

    /// Обратный проход Linear на GPU с использованием MatrixBufferHandle.
    /// Входные данные, веса, градиент выхода и все градиентные буферы должны быть GPU-буферами.
    /// `grad_weight` и `grad_bias_handle` будут обнулены перед накоплением.
    /// Возвращает градиент смещения как Vec<f32>.
    pub fn run_linear_backward_buffered_handle(
        &self,
        input: &MatrixBufferHandle,
        weight: &MatrixBufferHandle,
        grad_output: &MatrixBufferHandle,
        grad_input: &MatrixBufferHandle,
        grad_weight: &MatrixBufferHandle,
        grad_bias_handle: &MatrixBufferHandle,
    ) -> Vec<f32> {
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

        let x_buf = self.get_gpu_subbuffer_from_handle(input);
        let w_buf = self.get_gpu_subbuffer_from_handle(weight);
        let dout_buf = self.get_gpu_subbuffer_from_handle(grad_output);
        let dx_buf = self.get_gpu_subbuffer_from_handle(grad_input);
        let dw_buf = self.get_gpu_subbuffer_from_handle(grad_weight);
        let db_buf = self.get_gpu_subbuffer_from_handle(grad_bias_handle);

        // Обнуляем градиентные буферы перед атомарным накоплением
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

        // Скачиваем градиент смещения как Vec<f32>
        self.download_gpu_handle_to_vec(grad_bias_handle)
    }
}