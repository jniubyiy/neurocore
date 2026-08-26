// src/optimizers/add_weight_decay/gpu/mod.rs

pub mod pipeline;

use vulkano::buffer::Subbuffer;

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
        let pipeline = self.add_weight_decay_pipelines().forward.clone();
        self.run_compute_shader(
            pipeline,
            &[(0, params.clone()), (1, grads.clone())],
            &push,
            total,
        );
    }
}