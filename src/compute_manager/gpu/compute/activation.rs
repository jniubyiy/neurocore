// src/compute_manager/gpu/compute/activation.rs

use faer::Mat;
use vulkano::buffer::Subbuffer;
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::{Pipeline, PipelineBindPoint};
use super::base::GpuCompute;

impl GpuCompute {
    /// Прямой проход активации.
    /// Возвращает матрицу результата; все временные буферы возвращаются в пул.
    pub fn run_activation_forward(
        &self,
        input: &Mat<f32>,
        op_type: u32,
        alpha: f32,
    ) -> Mat<f32> {
        let batch = input.nrows();
        let features = input.ncols();
        let total_elements = batch * features;

        // Загружаем входные данные
        let (in_buf, in_raw) = self.upload_to_temp_buffer(&Self::mat_to_flat(input));

        // Выходной буфер
        let (out_buf, out_raw) = self.acquire_temp_buffer(total_elements);

        let pipeline = self.pipeline_cache.activation_pipeline();
        let set_layout = pipeline.layout().set_layouts().get(0).unwrap().clone();
        let descriptor_set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            set_layout.clone(),
            [
                WriteDescriptorSet::buffer(0, in_buf.clone()),
                WriteDescriptorSet::buffer(1, out_buf.clone()),
            ],
            [],
        )
        .expect("descriptor set");

        let push: [u32; 3] = [op_type, alpha.to_bits(), total_elements as u32];
        self.run_compute_shader(
            pipeline,
            &[(0, in_buf.clone()), (1, out_buf.clone())],
            &push,
            total_elements,
        );

        // Читаем результат и возвращаем буферы
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
        let set_layout = pipeline.layout().set_layouts().get(0).unwrap().clone();
        let descriptor_set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            set_layout.clone(),
            [
                WriteDescriptorSet::buffer(0, in_buf.clone()),
                WriteDescriptorSet::buffer(1, go_buf.clone()),
                WriteDescriptorSet::buffer(2, gi_buf.clone()),
            ],
            [],
        )
        .expect("descriptor set");

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
}