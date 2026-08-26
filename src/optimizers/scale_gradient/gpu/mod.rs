use vulkano::buffer::Subbuffer;

use crate::compute_manager::gpu::compute::GpuCompute;

impl GpuCompute {
    pub fn run_scale_gradient(&self, grads: &Subbuffer<[f32]>, factor: f32, total: usize) {
        let push = [factor.to_bits(), total as u32];
        self.run_compute_shader(
            self.pipeline_cache.scale_grad.clone(),
            &[(0, grads.clone())],
            &push,
            total,
        );
    }
}