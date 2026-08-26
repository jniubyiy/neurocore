// src/optimizers/momentum/gpu/mod.rs

use vulkano::buffer::Subbuffer;
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::Pipeline;

use crate::compute_manager::gpu::compute::GpuCompute;

impl GpuCompute {
    pub fn run_momentum(
        &self,
        grads: &Subbuffer<[f32]>,
        state: &Subbuffer<[f32]>,
        beta: f32,
        total: usize,
    ) {
        let push = [beta.to_bits(), total as u32];
        let set_layout = self
            .pipeline_cache
            .momentum
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
        .expect("momentum descriptor set");

        self.run_optimizer_with_ds(
            self.pipeline_cache.momentum.clone(),
            descriptor_set,
            &push,
        );
    }
}