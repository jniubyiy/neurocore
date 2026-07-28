// src/compute_manager/gpu/compute/activation.rs

use faer::Mat;
use vulkano::buffer::BufferUsage;
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage};
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::{Pipeline, PipelineBindPoint};
use vulkano::sync::{self, GpuFuture};
use super::base::GpuCompute;

impl GpuCompute {
    pub fn run_activation_forward(
        &self,
        input: &Mat<f32>,
        op_type: u32,
        alpha: f32,
    ) -> Mat<f32> {
        let batch = input.nrows();
        let features = input.ncols();
        let total_elements = batch * features;

        let (input_buf, input_id) = self.create_storage_buffer_from_slice(
            &(0..batch)
                .flat_map(|r| (0..features).map(move |c| input[(r, c)]))
                .collect::<Vec<f32>>(),
        );

        let (output_buf, output_id) = self.create_buffer(
            total_elements,
            BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC,
        );

        let pipeline = self.pipeline_cache.activation_pipeline();
        let set_layout = pipeline.layout().set_layouts().get(0).unwrap().clone();
        let descriptor_set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            set_layout.clone(),
            [
                WriteDescriptorSet::buffer(0, input_buf.clone()),
                WriteDescriptorSet::buffer(1, output_buf.clone()),
            ],
            [],
        )
        .expect("descriptor set");

        let mut builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.context.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .expect("command buffer builder");

        let push_constants: [u32; 3] = [op_type, alpha.to_bits(), total_elements as u32];

        unsafe {
            builder
                .bind_pipeline_compute(pipeline.clone())
                .unwrap()
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    pipeline.layout().clone(),
                    0,
                    descriptor_set,
                )
                .unwrap()
                .push_constants(pipeline.layout().clone(), 0, push_constants)
                .unwrap()
                .dispatch([((total_elements + 255) / 256) as u32, 1, 1])
                .unwrap();
        }

        let command_buffer = builder.build().expect("build command buffer");
        let future = sync::now(self.context.device.clone())
            .then_execute(self.context.queue.clone(), command_buffer)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();
        future.wait(None).unwrap();

        let mat = self.read_buffer_to_mat(output_buf, output_id, batch, features);
        self.release_buffer(input_id);
        mat
    }

    pub fn run_relu_forward(&self, input: &Mat<f32>) -> Mat<f32> { self.run_activation_forward(input, 0, 0.0) }
    pub fn run_sigmoid_forward(&self, input: &Mat<f32>) -> Mat<f32> { self.run_activation_forward(input, 1, 0.0) }
    pub fn run_tanh_forward(&self, input: &Mat<f32>) -> Mat<f32> { self.run_activation_forward(input, 2, 0.0) }
    pub fn run_leaky_relu_forward(&self, input: &Mat<f32>, alpha: f32) -> Mat<f32> { self.run_activation_forward(input, 3, alpha) }

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

        let (in_buf, in_id) = self.create_storage_buffer_from_slice(
            &(0..batch).flat_map(|r| (0..features).map(move |c| input_or_output[(r, c)])).collect::<Vec<f32>>(),
        );
        let (grad_out_buf, go_id) = self.create_storage_buffer_from_slice(
            &(0..batch).flat_map(|r| (0..features).map(move |c| grad_out[(r, c)])).collect::<Vec<f32>>(),
        );
        let (grad_in_buf, gi_id) = self.create_buffer(total_elements, BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC);

        let pipeline = self.pipeline_cache.activation_backward_pipeline();
        let set_layout = pipeline.layout().set_layouts().get(0).unwrap().clone();
        let descriptor_set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            set_layout.clone(),
            [
                WriteDescriptorSet::buffer(0, in_buf.clone()),
                WriteDescriptorSet::buffer(1, grad_out_buf.clone()),
                WriteDescriptorSet::buffer(2, grad_in_buf.clone()),
            ],
            [],
        )
        .expect("descriptor set");

        let mut builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.context.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .expect("command buffer builder");

        let push_constants: [u32; 3] = [op_type, alpha.to_bits(), total_elements as u32];

        unsafe {
            builder
                .bind_pipeline_compute(pipeline.clone())
                .unwrap()
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    pipeline.layout().clone(),
                    0,
                    descriptor_set,
                )
                .unwrap()
                .push_constants(pipeline.layout().clone(), 0, push_constants)
                .unwrap()
                .dispatch([((total_elements + 255) / 256) as u32, 1, 1])
                .unwrap();
        }

        let command_buffer = builder.build().expect("build command buffer");
        let future = sync::now(self.context.device.clone())
            .then_execute(self.context.queue.clone(), command_buffer)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();
        future.wait(None).unwrap();

        let mat = self.read_buffer_to_mat(grad_in_buf, gi_id, batch, features);
        self.release_buffer(in_id);
        self.release_buffer(go_id);
        mat
    }

    pub fn run_relu_backward(&self, input: &Mat<f32>, grad_output: &Mat<f32>) -> Mat<f32> { self.run_activation_backward(input, grad_output, 0, 0.0) }
    pub fn run_sigmoid_backward(&self, output: &Mat<f32>, grad_output: &Mat<f32>) -> Mat<f32> { self.run_activation_backward(output, grad_output, 1, 0.0) }
    pub fn run_tanh_backward(&self, output: &Mat<f32>, grad_output: &Mat<f32>) -> Mat<f32> { self.run_activation_backward(output, grad_output, 2, 0.0) }
    pub fn run_leaky_relu_backward(&self, input: &Mat<f32>, grad_output: &Mat<f32>, alpha: f32) -> Mat<f32> { self.run_activation_backward(input, grad_output, 3, alpha) }
}