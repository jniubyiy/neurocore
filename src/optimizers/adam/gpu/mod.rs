// src/optimizers/adam/gpu/mod.rs

use vulkano::buffer::Subbuffer;
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::Pipeline;

use crate::compute_manager::gpu::compute::GpuCompute;

impl GpuCompute {
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
        let set_layout = self
            .pipeline_cache
            .adam
            .layout()
            .set_layouts()
            .get(0)
            .unwrap()
            .clone();
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
}