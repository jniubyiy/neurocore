// src/optimizers/add_weight_decay/gpu/mod.rs

use vulkano::buffer::Subbuffer;
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::Pipeline;

use crate::compute_manager::gpu::compute::GpuCompute;

impl GpuCompute {
    pub fn run_weight_decay(
        &self,
        params: &Subbuffer<[f32]>,
        grads: &Subbuffer<[f32]>,
        decay: f32,
        total: usize,
    ) {
        let push = [decay.to_bits(), total as u32];
        let set_layout = self
            .pipeline_cache
            .weight_decay
            .layout()
            .set_layouts()
            .get(0)
            .unwrap()
            .clone();
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

        self.run_optimizer_with_ds(
            self.pipeline_cache.weight_decay.clone(),
            descriptor_set,
            &push,
        );
    }
}