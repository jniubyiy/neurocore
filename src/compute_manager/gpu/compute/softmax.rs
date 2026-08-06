// src/compute_manager/gpu/compute/softmax.rs

use faer::Mat;
use vulkano::buffer::BufferUsage;
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage};
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::{Pipeline, PipelineBindPoint};
use vulkano::sync::{self, GpuFuture};
use super::base::GpuCompute;

impl GpuCompute {
    pub fn run_softmax_forward(&self, input: &Mat<f32>) -> Mat<f32> {
        let batch = input.nrows();
        let cols = input.ncols();
        let total = batch * cols;

        let (input_buf, input_id) = self.create_storage_buffer_from_slice(
            &(0..batch).flat_map(|r| (0..cols).map(move |c| input[(r, c)])).collect::<Vec<f32>>(),
        );
        let (output_buf, output_id) = self.create_buffer(total, BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC);

        let pipeline = self.pipeline_cache.softmax_pipeline();
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

        let push: [u32; 2] = [batch as u32, cols as u32];

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
                .push_constants(pipeline.layout().clone(), 0, push)
                .unwrap()
                .dispatch([batch as u32, 1, 1])
                .unwrap();
        }

        let command_buffer = builder.build().expect("build command buffer");
        let future = sync::now(self.context.device.clone())
            .then_execute(self.context.queue.clone(), command_buffer)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();
        future.wait(None).unwrap();

        let mat = self.read_buffer_to_mat(output_id, batch, cols);
        self.release_buffer(input_id);
        mat
    }

    pub fn run_softmax_backward(&self, output: &Mat<f32>, grad_output: &Mat<f32>) -> Mat<f32> {
        let batch = output.nrows();
        let cols = output.ncols();
        let total = batch * cols;

        let (y_buf, y_id) = self.create_storage_buffer_from_slice(
            &(0..batch).flat_map(|r| (0..cols).map(move |c| output[(r, c)])).collect::<Vec<f32>>(),
        );
        let (grad_out_buf, go_id) = self.create_storage_buffer_from_slice(
            &(0..batch).flat_map(|r| (0..cols).map(move |c| grad_output[(r, c)])).collect::<Vec<f32>>(),
        );
        let (grad_in_buf, gi_id) = self.create_buffer(total, BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC);

        let pipeline = self.pipeline_cache.softmax_backward_pipeline();
        let set_layout = pipeline.layout().set_layouts().get(0).unwrap().clone();
        let descriptor_set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            set_layout.clone(),
            [
                WriteDescriptorSet::buffer(0, y_buf.clone()),
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

        let push: [u32; 2] = [batch as u32, cols as u32];

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
                .push_constants(pipeline.layout().clone(), 0, push)
                .unwrap()
                .dispatch([batch as u32, 1, 1])
                .unwrap();
        }

        let command_buffer = builder.build().expect("build command buffer");
        let future = sync::now(self.context.device.clone())
            .then_execute(self.context.queue.clone(), command_buffer)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();
        future.wait(None).unwrap();

        let mat = self.read_buffer_to_mat(gi_id, batch, cols);
        self.release_buffer(y_id);
        self.release_buffer(go_id);
        mat
    }
}