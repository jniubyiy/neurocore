// src/compute_manager/gpu/compute/activation.rs

use faer::Mat;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo};
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};
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

        let input_buf = Self::create_storage_buffer_from_slice(
            &self.context.memory_allocator,
            &(0..batch)
                .flat_map(|r| (0..features).map(move |c| input[(r, c)]))
                .collect::<Vec<f32>>(),
        );

        let output_size = total_elements as u64 * std::mem::size_of::<f32>() as u64;
        let output_buf: Subbuffer<[f32]> = Buffer::new_unsized(
            self.context.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            output_size,
        )
        .expect("output buffer");

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

        let staging_size = total_elements as u64 * std::mem::size_of::<f32>() as u64;
        let staging_buf: Subbuffer<[f32]> = Buffer::new_unsized(
            self.context.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            staging_size,
        )
        .expect("staging buffer");

        let mut copy_builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.context.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();
        copy_builder
            .copy_buffer(CopyBufferInfo::buffers(output_buf, staging_buf.clone()))
            .unwrap();
        let copy_cb = copy_builder.build().unwrap();
        let future2 = sync::now(self.context.device.clone())
            .then_execute(self.context.queue.clone(), copy_cb)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();
        future2.wait(None).unwrap();

        let data = staging_buf.read().expect("read staging buffer");
        Mat::from_fn(batch, features, |r, c| data[r * features + c])
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

        let in_buf = Self::create_storage_buffer_from_slice(
            &self.context.memory_allocator,
            &(0..batch).flat_map(|r| (0..features).map(move |c| input_or_output[(r, c)])).collect::<Vec<f32>>(),
        );
        let grad_out_buf = Self::create_storage_buffer_from_slice(
            &self.context.memory_allocator,
            &(0..batch).flat_map(|r| (0..features).map(move |c| grad_out[(r, c)])).collect::<Vec<f32>>(),
        );
        let grad_in_buf = self.create_buffer(total_elements, BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC);

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

        let staging = self.create_buffer(total_elements, BufferUsage::TRANSFER_DST);
        self.copy_buffer_sync(grad_in_buf, staging.clone());
        let data = staging.read().unwrap();
        Mat::from_fn(batch, features, |r, c| data[r * features + c])
    }

    pub fn run_relu_backward(&self, input: &Mat<f32>, grad_output: &Mat<f32>) -> Mat<f32> { self.run_activation_backward(input, grad_output, 0, 0.0) }
    pub fn run_sigmoid_backward(&self, output: &Mat<f32>, grad_output: &Mat<f32>) -> Mat<f32> { self.run_activation_backward(output, grad_output, 1, 0.0) }
    pub fn run_tanh_backward(&self, output: &Mat<f32>, grad_output: &Mat<f32>) -> Mat<f32> { self.run_activation_backward(output, grad_output, 2, 0.0) }
    pub fn run_leaky_relu_backward(&self, input: &Mat<f32>, grad_output: &Mat<f32>, alpha: f32) -> Mat<f32> { self.run_activation_backward(input, grad_output, 3, alpha) }
}