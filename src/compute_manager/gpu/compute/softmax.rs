// src/compute_manager/gpu/compute/softmax.rs

use faer::Mat;
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::{Pipeline, PipelineBindPoint};
use super::base::GpuCompute;
use crate::compute_manager::matrix_buffer::MatrixBuffer;

impl GpuCompute {
    /// Прямой проход softmax на GPU.
    /// Старая версия для обратной совместимости: принимает `Mat`, возвращает `Mat`.
    pub fn run_softmax_forward(&self, input: &Mat<f32>) -> Mat<f32> {
        let batch = input.nrows();
        let cols = input.ncols();
        let total = batch * cols;

        let flat_input = Self::mat_to_flat(input);
        let (in_buf, in_raw) = self.upload_to_temp_buffer(&flat_input);
        let (out_buf, out_raw) = self.acquire_temp_buffer(total);

        let pipeline = self.pipeline_cache.softmax_pipeline();
        let push: [u32; 2] = [batch as u32, cols as u32];
        self.run_compute_shader_with_dispatch(
            pipeline,
            &[(0, in_buf.clone()), (1, out_buf.clone())],
            &push,
            [batch as u32, 1, 1],      // <-- правильный диспатч
        );

        let result = self.read_temp_buffer_to_mat(out_buf, out_raw, batch, cols);
        self.release_temp_buffer(in_buf, in_raw);
        result
    }

    /// Обратный проход softmax на GPU.
    /// Старая версия для обратной совместимости.
    pub fn run_softmax_backward(&self, output: &Mat<f32>, grad_output: &Mat<f32>) -> Mat<f32> {
        let batch = output.nrows();
        let cols = output.ncols();
        let total = batch * cols;

        let (y_buf, y_raw) = self.upload_to_temp_buffer(&Self::mat_to_flat(output));
        let (go_buf, go_raw) = self.upload_to_temp_buffer(&Self::mat_to_flat(grad_output));
        let (gi_buf, gi_raw) = self.acquire_temp_buffer(total);

        let pipeline = self.pipeline_cache.softmax_backward_pipeline();
        let push: [u32; 2] = [batch as u32, cols as u32];
        self.run_compute_shader_with_dispatch(
            pipeline,
            &[(0, y_buf.clone()), (1, go_buf.clone()), (2, gi_buf.clone())],
            &push,
            [batch as u32, 1, 1],      // <-- правильный диспатч
        );

        let dx = self.read_temp_buffer_to_mat(gi_buf, gi_raw, batch, cols);
        self.release_temp_buffer(y_buf, y_raw);
        self.release_temp_buffer(go_buf, go_raw);
        dx
    }

    // ===================================================================
    // НОВЫЕ БУФЕРИЗОВАННЫЕ ВЕРСИИ (MatrixBuffer)
    // ===================================================================

    /// Прямой проход softmax на GPU с использованием MatrixBuffer.
    /// Принимает входной GPU-буфер и возвращает выходной GPU-буфер.
    pub fn run_softmax_forward_buffered(&self, input: &MatrixBuffer) -> MatrixBuffer {
        assert!(input.is_gpu(), "Input buffer must be GPU");
        let batch = input.rows();
        let cols = input.cols();
        let total = batch * cols;

        let in_buf = input.as_gpu_buffer().expect("GPU buffer");
        let out = self.allocate_gpu_matrix(batch, cols);
        let out_buf = out.as_gpu_buffer().expect("GPU buffer");

        let pipeline = self.pipeline_cache.softmax_pipeline();
        let push: [u32; 2] = [batch as u32, cols as u32];

        self.run_compute_shader_with_dispatch(
            pipeline,
            &[(0, in_buf.clone()), (1, out_buf.clone())],
            &push,
            [batch as u32, 1, 1],
        );

        out
    }

    /// Обратный проход softmax на GPU с использованием MatrixBuffer.
    /// Принимает выход softmax (GPU) и градиент по выходу (GPU), возвращает градиент по входу (GPU).
    pub fn run_softmax_backward_buffered(
        &self,
        output: &MatrixBuffer,
        grad_output: &MatrixBuffer,
    ) -> MatrixBuffer {
        assert!(output.is_gpu() && grad_output.is_gpu(), "Buffers must be GPU");
        let batch = output.rows();
        let cols = output.cols();
        let total = batch * cols;
        assert_eq!(grad_output.rows(), batch);
        assert_eq!(grad_output.cols(), cols);

        let y_buf = output.as_gpu_buffer().expect("GPU buffer");
        let go_buf = grad_output.as_gpu_buffer().expect("GPU buffer");
        let gi = self.allocate_gpu_matrix(batch, cols);
        let gi_buf = gi.as_gpu_buffer().expect("GPU buffer");

        let pipeline = self.pipeline_cache.softmax_backward_pipeline();
        let push: [u32; 2] = [batch as u32, cols as u32];

        self.run_compute_shader_with_dispatch(
            pipeline,
            &[(0, y_buf.clone()), (1, go_buf.clone()), (2, gi_buf.clone())],
            &push,
            [batch as u32, 1, 1],
        );

        gi
    }
}