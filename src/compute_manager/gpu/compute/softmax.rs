// src/compute_manager/gpu/compute/softmax.rs

use super::base::GpuCompute;
use crate::compute_manager::matrix_buffer::MatrixBufferHandle;

impl GpuCompute {
    // ===================================================================
    // Handle-версии (MatrixBufferHandle)
    // ===================================================================

    /// Прямой проход softmax на GPU с использованием MatrixBufferHandle.
    /// Вход и выход должны быть GPU-буферами.
    pub fn run_softmax_forward_buffered_handle(
        &self,
        input: &MatrixBufferHandle,
        output: &MatrixBufferHandle,
    ) {
        assert!(input.is_gpu(), "Input handle must be GPU");
        assert!(output.is_gpu(), "Output handle must be GPU");

        let batch = input.rows();
        let cols = input.cols();
        assert_eq!(batch, output.rows(), "Batch mismatch");
        assert_eq!(cols, output.cols(), "Column mismatch");

        let in_buf = self.get_gpu_subbuffer_from_handle(input);
        let out_buf = self.get_gpu_subbuffer_from_handle(output);

        let pipeline = self.pipeline_cache.softmax_pipeline();
        let push: [u32; 2] = [batch as u32, cols as u32];

        self.run_compute_shader_with_dispatch(
            pipeline,
            &[(0, in_buf), (1, out_buf)],
            &push,
            [batch as u32, 1, 1],
        );
    }

    /// Обратный проход softmax на GPU с использованием MatrixBufferHandle.
    /// `output` — выход softmax (GPU), `grad_output` — градиент по выходу (GPU),
    /// `grad_input` — буфер для записи градиента по входу (GPU).
    pub fn run_softmax_backward_buffered_handle(
        &self,
        output: &MatrixBufferHandle,
        grad_output: &MatrixBufferHandle,
        grad_input: &MatrixBufferHandle,
    ) {
        assert!(output.is_gpu(), "Output handle must be GPU");
        assert!(grad_output.is_gpu(), "grad_output handle must be GPU");
        assert!(grad_input.is_gpu(), "grad_input handle must be GPU");

        let batch = output.rows();
        let cols = output.cols();
        assert_eq!(grad_output.rows(), batch);
        assert_eq!(grad_output.cols(), cols);
        assert_eq!(grad_input.rows(), batch);
        assert_eq!(grad_input.cols(), cols);

        let y_buf = self.get_gpu_subbuffer_from_handle(output);
        let go_buf = self.get_gpu_subbuffer_from_handle(grad_output);
        let gi_buf = self.get_gpu_subbuffer_from_handle(grad_input);

        let pipeline = self.pipeline_cache.softmax_backward_pipeline();
        let push: [u32; 2] = [batch as u32, cols as u32];

        self.run_compute_shader_with_dispatch(
            pipeline,
            &[(0, y_buf), (1, go_buf), (2, gi_buf)],
            &push,
            [batch as u32, 1, 1],
        );
    }
}