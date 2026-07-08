// src/compute_manager/gpu/compute/optimizer.rs

use std::sync::Arc;
use vulkano::buffer::Subbuffer;
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage};
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::{Pipeline, PipelineBindPoint};
use vulkano::sync::{self, GpuFuture};
use super::base::GpuCompute;

impl GpuCompute {
    pub fn run_scale_gradient(&self, grads: &Subbuffer<[f32]>, factor: f32, total: usize) {
        let push = [factor.to_bits(), total as u32];
        self.run_optimizer_1buf(self.pipeline_cache.scale_grad.clone(), grads, &push);
    }

    pub fn run_weight_decay(
        &self,
        params: &Subbuffer<[f32]>,
        grads: &Subbuffer<[f32]>,
        decay: f32,
        total: usize,
    ) {
        let push = [decay.to_bits(), total as u32];
        let set_layout = self.pipeline_cache.weight_decay.layout().set_layouts().get(0).unwrap().clone();
        let descriptor_set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            set_layout.clone(),
            [
                WriteDescriptorSet::buffer(0, params.clone()),
                WriteDescriptorSet::buffer(1, grads.clone()),
            ],
            [],
        )
        .expect("weight_decay descriptor set");

        self.run_optimizer_with_ds(self.pipeline_cache.weight_decay.clone(), descriptor_set, &push);
    }

    pub fn run_gradient_clip(
        &self,
        grads: &Subbuffer<[f32]>,
        min_val: f32,
        max_val: f32,
        total: usize,
    ) {
        let push = [min_val.to_bits(), max_val.to_bits(), total as u32];
        self.run_optimizer_1buf(self.pipeline_cache.grad_clip.clone(), grads, &push);
    }

    pub fn run_momentum(
        &self,
        grads: &Subbuffer<[f32]>,
        state: &Subbuffer<[f32]>,
        beta: f32,
        total: usize,
    ) {
        let push = [beta.to_bits(), total as u32];
        let set_layout = self.pipeline_cache.momentum.layout().set_layouts().get(0).unwrap().clone();
        let descriptor_set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            set_layout.clone(),
            [
                WriteDescriptorSet::buffer(0, grads.clone()),
                WriteDescriptorSet::buffer(1, state.clone()),
            ],
            [],
        )
        .expect("momentum descriptor set");

        self.run_optimizer_with_ds(self.pipeline_cache.momentum.clone(), descriptor_set, &push);
    }

    pub fn run_nesterov_momentum(
        &self,
        grads: &Subbuffer<[f32]>,
        state: &Subbuffer<[f32]>,
        beta: f32,
        total: usize,
    ) {
        let push = [beta.to_bits(), total as u32];
        let set_layout = self.pipeline_cache.nesterov_momentum.layout().set_layouts().get(0).unwrap().clone();
        let descriptor_set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            set_layout.clone(),
            [
                WriteDescriptorSet::buffer(0, grads.clone()),
                WriteDescriptorSet::buffer(1, state.clone()),
            ],
            [],
        )
        .expect("nesterov_momentum descriptor set");

        self.run_optimizer_with_ds(self.pipeline_cache.nesterov_momentum.clone(), descriptor_set, &push);
    }

    pub fn run_adam(
        &self,
        grads: &Subbuffer<[f32]>,
        state: &Subbuffer<[f32]>,
        beta1: f32,
        beta2: f32,
        eps: f32,
        step: usize,
        total: usize,
    ) {
        let bias_correction1 = 1.0f32 - beta1.powi(step as i32);
        let bias_correction2 = 1.0f32 - beta2.powi(step as i32);
        let push = [
            beta1.to_bits(),
            beta2.to_bits(),
            eps.to_bits(),
            bias_correction1.to_bits(),
            bias_correction2.to_bits(),
            total as u32,
        ];
        let set_layout = self.pipeline_cache.adam.layout().set_layouts().get(0).unwrap().clone();
        let descriptor_set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            set_layout.clone(),
            [
                WriteDescriptorSet::buffer(0, grads.clone()),
                WriteDescriptorSet::buffer(1, state.clone()),
            ],
            [],
        )
        .expect("adam descriptor set");

        self.run_optimizer_with_ds(self.pipeline_cache.adam.clone(), descriptor_set, &push);
    }

    pub fn run_apply_update(
        &self,
        params: &Subbuffer<[f32]>,
        grads: &Subbuffer<[f32]>,
        total: usize,
    ) {
        let push = [total as u32];
        let set_layout = self.pipeline_cache.apply_update.layout().set_layouts().get(0).unwrap().clone();
        let descriptor_set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            set_layout.clone(),
            [
                WriteDescriptorSet::buffer(0, params.clone()),
                WriteDescriptorSet::buffer(1, grads.clone()),
            ],
            [],
        )
        .expect("apply_update descriptor set");

        self.run_optimizer_with_ds(self.pipeline_cache.apply_update.clone(), descriptor_set, &push);
    }

    fn run_optimizer_1buf<const N: usize>(
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

    fn run_optimizer_with_ds<const N: usize>(
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