// src/optimizers/apply_update/gpu/mod.rs

pub mod pipeline;

use vulkano::buffer::Subbuffer;

use crate::compute_manager::gpu::compute::GpuCompute;

impl GpuCompute {
    pub fn run_apply_update(
        &self,
        params: &Subbuffer<[f32]>,
        grads: &Subbuffer<[f32]>,
        total: usize,
    ) {
        let push = [total as u32];
        // Используем ссылку на Arc, чтобы избежать клонирования.
        let pipeline = &self.apply_update_pipelines().forward;
        self.run_compute_shader(
            pipeline,
            &[(0, params.clone()), (1, grads.clone())],
            &push,
            total,
        );
    }
}