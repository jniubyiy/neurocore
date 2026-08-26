// src/optimizers/momentum/gpu/mod.rs

pub mod pipeline;

use vulkano::buffer::Subbuffer;

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
        let pipeline = self.momentum_pipelines().forward.clone();
        self.run_compute_shader(
            pipeline,
            &[(0, grads.clone()), (1, state.clone())],
            &push,
            total,
        );
    }
}