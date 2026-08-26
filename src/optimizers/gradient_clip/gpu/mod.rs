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
        self.run_optimizer_1buf(self.pipeline_cache.grad_clip.clone(), grads, &push);
    }
}