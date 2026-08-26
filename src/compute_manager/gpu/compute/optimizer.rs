// src/compute_manager/gpu/compute/optimizer.rs

use std::sync::Arc;
use vulkano::buffer::Subbuffer;
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage};
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::{Pipeline, PipelineBindPoint};
use vulkano::sync::{self, GpuFuture};
use super::base::GpuCompute;

impl GpuCompute {
    pub fn run_optimizer_1buf<const N: usize>(
        &self,
        pipeline: Arc<vulkano::pipeline::ComputePipeline>,
        buf: &Subbuffer<[f32]>,
        push: &[u32; N],
    ) {
        let set_layout = pipeline.layout().set_layouts().get(0).unwrap().clone();
        let descriptor_set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            set_layout.clone(),
            [WriteDescriptorSet::buffer(0, buf.clone())],
            [],
        )
        .expect("optimizer 1‑buf descriptor set");

        self.run_optimizer_with_ds(pipeline, descriptor_set, push);
    }

    pub fn run_optimizer_with_ds<const N: usize>(
        &self,
        pipeline: Arc<vulkano::pipeline::ComputePipeline>,
        descriptor_set: Arc<DescriptorSet>,
        push: &[u32; N],
    ) {
        let mut builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.context.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .expect("optimizer command buffer builder");

        let total_elements = push.last().copied().unwrap_or(1);
        let dispatch_dim = [((total_elements + 255) / 256) as u32, 1, 1];

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
                .push_constants(pipeline.layout().clone(), 0, *push)
                .unwrap()
                .dispatch(dispatch_dim)
                .unwrap();
        }

        let command_buffer = builder.build().expect("build optimizer command buffer");
        let future = sync::now(self.context.device.clone())
            .then_execute(self.context.queue.clone(), command_buffer)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();
        future.wait(None).unwrap();
    }
}