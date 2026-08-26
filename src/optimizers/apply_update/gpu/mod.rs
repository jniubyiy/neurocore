// src/optimizers/apply_update/gpu/mod.rs

use vulkano::buffer::Subbuffer;
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::Pipeline;

use crate::compute_manager::gpu::compute::GpuCompute;

impl GpuCompute {
    pub fn run_apply_update(
        &self,
        params: &Subbuffer<[f32]>,
        grads: &Subbuffer<[f32]>,
        total: usize,
    ) {
        let push = [total as u32];
        let set_layout = self
            .pipeline_cache
            .apply_update
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
        .expect("apply_update descriptor set");

        self.run_optimizer_with_ds(
            self.pipeline_cache.apply_update.clone(),
            descriptor_set,
            &push,
        );
    }
}