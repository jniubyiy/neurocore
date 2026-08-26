// src/optimizers/gradient_clip/gpu/mod.rs

pub mod pipeline;

use vulkano::buffer::Subbuffer;

use crate::compute_manager::gpu::compute::GpuCompute;

impl GpuCompute {
    pub fn run_gradient_clip(
        &self,
        grads: &Subbuffer<[f32]>,
        min_val: f32,
        max_val: f32,
        total: usize,
    ) {
        let push = [min_val.to_bits(), max_val.to_bits(), total as u32];
        let pipeline = self.gradient_clip_pipelines().forward.clone();
        self.run_compute_shader(
            pipeline,
            &[(0, grads.clone())],
            &push,
            total,
        );
    }
}