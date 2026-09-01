// src/optimizers/adam/gpu/mod.rs

pub mod pipeline;

use vulkano::buffer::Subbuffer;

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
        // Используем ссылку на Arc, чтобы избежать клонирования.
        let pipeline = &self.adam_pipelines().forward;
        self.run_compute_shader(
            pipeline,
            &[(0, grads.clone()), (1, state.clone())],
            &push,
            total,
        );
    }
}