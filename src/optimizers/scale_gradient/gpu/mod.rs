// src/optimizers/scale_gradient/gpu/mod.rs

pub mod pipeline;

use vulkano::buffer::Subbuffer;

use crate::compute_manager::gpu::compute::GpuCompute;

impl GpuCompute {
    pub fn run_scale_gradient(
        &self,
        grads: &Subbuffer<[f32]>,
        factor: f32,
        total: usize,
    ) {
        let push = [factor.to_bits(), total as u32];
        // Используем ссылку на Arc, чтобы избежать клонирования.
        let pipeline = &self.scale_gradient_pipelines().forward;
        self.run_compute_shader(
            pipeline,
            &[(0, grads.clone())],
            &push,
            total,
        );
    }
}