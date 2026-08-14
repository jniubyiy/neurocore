// src/compute_manager/gpu/compute/activation.rs

use faer::Mat;
use vulkano::buffer::Subbuffer;
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::{Pipeline, PipelineBindPoint};
use super::base::GpuCompute;
use crate::compute_manager::matrix_buffer::MatrixBuffer;
use crate::compute_manager::matrix_buffer::MatrixBufferHandle;

impl GpuCompute {
    // ===================================================================
    // Старые Mat-версии (оставлены для обратной совместимости)
    // ===================================================================

    /// Прямой проход активации.
    pub fn run_activation_forward(
        &self,
        input: &Mat<f32>,
        op_type: u32,
        alpha: f32,
    ) -> Mat<f32> {
        let batch = input.nrows();
        let features = input.ncols();
        let total_elements = batch * features;

        let (in_buf, in_raw) = self.upload_to_temp_buffer(&Self::mat_to_flat(input));
        let (out_buf, out_raw) = self.acquire_temp_buffer(total_elements);

        let pipeline = self.pipeline_cache.activation_pipeline();
        let push: [u32; 3] = [op_type, alpha.to_bits(), total_elements as u32];
        self.run_compute_shader(
            pipeline,
            &[(0, in_buf.clone()), (1, out_buf.clone())],
            &push,
            total_elements,
        );

        let result = self.read_temp_buffer_to_mat(out_buf, out_raw, batch, features);
        self.release_temp_buffer(in_buf, in_raw);

        result
    }

    pub fn run_relu_forward(&self, input: &Mat<f32>) -> Mat<f32> {
        self.run_activation_forward(input, 0, 0.0)
    }
    pub fn run_sigmoid_forward(&self, input: &Mat<f32>) -> Mat<f32> {
        self.run_activation_forward(input, 1, 0.0)
    }
    pub fn run_tanh_forward(&self, input: &Mat<f32>) -> Mat<f32> {
        self.run_activation_forward(input, 2, 0.0)
    }
    pub fn run_leaky_relu_forward(&self, input: &Mat<f32>, alpha: f32) -> Mat<f32> {
        self.run_activation_forward(input, 3, alpha)
    }

    /// Обратный проход активации.
    pub fn run_activation_backward(
        &self,
        input_or_output: &Mat<f32>,
        grad_out: &Mat<f32>,
        op_type: u32,
        alpha: f32,
    ) -> Mat<f32> {
        let batch = input_or_output.nrows();
        let features = input_or_output.ncols();
        let total_elements = batch * features;
        assert_eq!(grad_out.nrows(), batch);
        assert_eq!(grad_out.ncols(), features);

        let (in_buf, in_raw) = self.upload_to_temp_buffer(&Self::mat_to_flat(input_or_output));
        let (go_buf, go_raw) = self.upload_to_temp_buffer(&Self::mat_to_flat(grad_out));
        let (gi_buf, gi_raw) = self.acquire_temp_buffer(total_elements);

        let pipeline = self.pipeline_cache.activation_backward_pipeline();
        let push: [u32; 3] = [op_type, alpha.to_bits(), total_elements as u32];
        self.run_compute_shader(
            pipeline,
            &[(0, in_buf.clone()), (1, go_buf.clone()), (2, gi_buf.clone())],
            &push,
            total_elements,
        );

        let result = self.read_temp_buffer_to_mat(gi_buf, gi_raw, batch, features);
        self.release_temp_buffer(in_buf, in_raw);
        self.release_temp_buffer(go_buf, go_raw);

        result
    }

    pub fn run_relu_backward(&self, input: &Mat<f32>, grad_output: &Mat<f32>) -> Mat<f32> {
        self.run_activation_backward(input, grad_output, 0, 0.0)
    }
    pub fn run_sigmoid_backward(&self, output: &Mat<f32>, grad_output: &Mat<f32>) -> Mat<f32> {
        self.run_activation_backward(output, grad_output, 1, 0.0)
    }
    pub fn run_tanh_backward(&self, output: &Mat<f32>, grad_output: &Mat<f32>) -> Mat<f32> {
        self.run_activation_backward(output, grad_output, 2, 0.0)
    }
    pub fn run_leaky_relu_backward(&self, input: &Mat<f32>, grad_output: &Mat<f32>, alpha: f32) -> Mat<f32> {
        self.run_activation_backward(input, grad_output, 3, alpha)
    }

    // ===================================================================
    // Буферизованные версии (MatrixBuffer)
    // ===================================================================

    /// Прямой проход активации на GPU с использованием MatrixBuffer.
    pub fn run_activation_forward_buffered(
        &self,
        input: &MatrixBuffer,
        op_type: u32,
        alpha: f32,
    ) -> MatrixBuffer {
        assert!(input.is_gpu(), "Input buffer must be GPU");
        let rows = input.rows();
        let cols = input.cols();
        let total_elements = rows * cols;

        let in_buf = input.as_gpu_buffer().expect("GPU buffer");
        let out = self.allocate_gpu_matrix(rows, cols);
        let out_buf = out.as_gpu_buffer().expect("GPU buffer");

        let pipeline = self.pipeline_cache.activation_pipeline();
        let push: [u32; 3] = [op_type, alpha.to_bits(), total_elements as u32];

        self.run_compute_shader(
            pipeline,
            &[(0, in_buf.clone()), (1, out_buf.clone())],
            &push,
            total_elements,
        );

        out
    }

    pub fn run_relu_forward_buffered(&self, input: &MatrixBuffer) -> MatrixBuffer {
        self.run_activation_forward_buffered(input, 0, 0.0)
    }
    pub fn run_sigmoid_forward_buffered(&self, input: &MatrixBuffer) -> MatrixBuffer {
        self.run_activation_forward_buffered(input, 1, 0.0)
    }
    pub fn run_tanh_forward_buffered(&self, input: &MatrixBuffer) -> MatrixBuffer {
        self.run_activation_forward_buffered(input, 2, 0.0)
    }
    pub fn run_leaky_relu_forward_buffered(&self, input: &MatrixBuffer, alpha: f32) -> MatrixBuffer {
        self.run_activation_forward_buffered(input, 3, alpha)
    }

    /// Обратный проход активации на GPU с использованием MatrixBuffer.
    pub fn run_activation_backward_buffered(
        &self,
        input_or_output: &MatrixBuffer,
        grad_out: &MatrixBuffer,
        op_type: u32,
        alpha: f32,
    ) -> MatrixBuffer {
        assert!(input_or_output.is_gpu() && grad_out.is_gpu(), "Buffers must be GPU");
        let rows = input_or_output.rows();
        let cols = input_or_output.cols();
        let total_elements = rows * cols;
        assert_eq!(grad_out.rows(), rows);
        assert_eq!(grad_out.cols(), cols);

        let in_buf = input_or_output.as_gpu_buffer().expect("GPU buffer");
        let go_buf = grad_out.as_gpu_buffer().expect("GPU buffer");
        let gi = self.allocate_gpu_matrix(rows, cols);
        let gi_buf = gi.as_gpu_buffer().expect("GPU buffer");

        let pipeline = self.pipeline_cache.activation_backward_pipeline();
        let push: [u32; 3] = [op_type, alpha.to_bits(), total_elements as u32];

        self.run_compute_shader(
            pipeline,
            &[(0, in_buf.clone()), (1, go_buf.clone()), (2, gi_buf.clone())],
            &push,
            total_elements,
        );

        gi
    }

    pub fn run_relu_backward_buffered(&self, input: &MatrixBuffer, grad_output: &MatrixBuffer) -> MatrixBuffer {
        self.run_activation_backward_buffered(input, grad_output, 0, 0.0)
    }
    pub fn run_sigmoid_backward_buffered(&self, output: &MatrixBuffer, grad_output: &MatrixBuffer) -> MatrixBuffer {
        self.run_activation_backward_buffered(output, grad_output, 1, 0.0)
    }
    pub fn run_tanh_backward_buffered(&self, output: &MatrixBuffer, grad_output: &MatrixBuffer) -> MatrixBuffer {
        self.run_activation_backward_buffered(output, grad_output, 2, 0.0)
    }
    pub fn run_leaky_relu_backward_buffered(&self, input: &MatrixBuffer, grad_output: &MatrixBuffer, alpha: f32) -> MatrixBuffer {
        self.run_activation_backward_buffered(input, grad_output, 3, alpha)
    }

    // ===================================================================
    // НОВЫЕ Handle-версии (MatrixBufferHandle)
    // ===================================================================

    /// Прямой проход активации на GPU с использованием MatrixBufferHandle.
    /// Вход и выход должны быть GPU-буферами.
    pub fn run_activation_forward_buffered_handle(
        &self,
        input: &MatrixBufferHandle,
        output: &MatrixBufferHandle,
        op_type: u32,
        alpha: f32,
    ) {
        assert!(input.is_gpu() && output.is_gpu(), "Handles must be GPU");
        let total_elements = input.rows() * input.cols();
        assert_eq!(total_elements, output.rows() * output.cols(), "Shape mismatch");

        let in_buf = self.get_gpu_subbuffer_from_handle(input);
        let out_buf = self.get_gpu_subbuffer_from_handle(output);

        let pipeline = self.pipeline_cache.activation_pipeline();
        let push: [u32; 3] = [op_type, alpha.to_bits(), total_elements as u32];

        self.run_compute_shader(
            pipeline,
            &[(0, in_buf), (1, out_buf)],
            &push,
            total_elements,
        );
    }

    pub fn run_relu_forward_buffered_handle(
        &self,
        input: &MatrixBufferHandle,
        output: &MatrixBufferHandle,
    ) {
        self.run_activation_forward_buffered_handle(input, output, 0, 0.0)
    }
    pub fn run_sigmoid_forward_buffered_handle(
        &self,
        input: &MatrixBufferHandle,
        output: &MatrixBufferHandle,
    ) {
        self.run_activation_forward_buffered_handle(input, output, 1, 0.0)
    }
    pub fn run_tanh_forward_buffered_handle(
        &self,
        input: &MatrixBufferHandle,
        output: &MatrixBufferHandle,
    ) {
        self.run_activation_forward_buffered_handle(input, output, 2, 0.0)
    }
    pub fn run_leaky_relu_forward_buffered_handle(
        &self,
        input: &MatrixBufferHandle,
        output: &MatrixBufferHandle,
        alpha: f32,
    ) {
        self.run_activation_forward_buffered_handle(input, output, 3, alpha)
    }

    /// Обратный проход активации на GPU с использованием MatrixBufferHandle.
    /// Вход/выход (в зависимости от операции) и градиенты должны быть GPU-буферами.
    pub fn run_activation_backward_buffered_handle(
        &self,
        input_or_output: &MatrixBufferHandle,
        grad_out: &MatrixBufferHandle,
        grad_input: &MatrixBufferHandle,
        op_type: u32,
        alpha: f32,
    ) {
        assert!(input_or_output.is_gpu(), "input_or_output must be GPU");
        assert!(grad_out.is_gpu(), "grad_out must be GPU");
        assert!(grad_input.is_gpu(), "grad_input must be GPU");

        let total_elements = input_or_output.rows() * input_or_output.cols();
        assert_eq!(total_elements, grad_out.rows() * grad_out.cols(), "grad_out shape mismatch");
        assert_eq!(total_elements, grad_input.rows() * grad_input.cols(), "grad_input shape mismatch");

        let in_buf = self.get_gpu_subbuffer_from_handle(input_or_output);
        let go_buf = self.get_gpu_subbuffer_from_handle(grad_out);
        let gi_buf = self.get_gpu_subbuffer_from_handle(grad_input);

        let pipeline = self.pipeline_cache.activation_backward_pipeline();
        let push: [u32; 3] = [op_type, alpha.to_bits(), total_elements as u32];

        self.run_compute_shader(
            pipeline,
            &[(0, in_buf), (1, go_buf), (2, gi_buf)],
            &push,
            total_elements,
        );
    }

    pub fn run_relu_backward_buffered_handle(
        &self,
        input: &MatrixBufferHandle,
        grad_output: &MatrixBufferHandle,
        grad_input: &MatrixBufferHandle,
    ) {
        self.run_activation_backward_buffered_handle(input, grad_output, grad_input, 0, 0.0)
    }
    pub fn run_sigmoid_backward_buffered_handle(
        &self,
        output: &MatrixBufferHandle,
        grad_output: &MatrixBufferHandle,
        grad_input: &MatrixBufferHandle,
    ) {
        self.run_activation_backward_buffered_handle(output, grad_output, grad_input, 1, 0.0)
    }
    pub fn run_tanh_backward_buffered_handle(
        &self,
        output: &MatrixBufferHandle,
        grad_output: &MatrixBufferHandle,
        grad_input: &MatrixBufferHandle,
    ) {
        self.run_activation_backward_buffered_handle(output, grad_output, grad_input, 2, 0.0)
    }
    pub fn run_leaky_relu_backward_buffered_handle(
        &self,
        input: &MatrixBufferHandle,
        grad_output: &MatrixBufferHandle,
        grad_input: &MatrixBufferHandle,
        alpha: f32,
    ) {
        self.run_activation_backward_buffered_handle(input, grad_output, grad_input, 3, alpha)
    }
}