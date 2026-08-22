// src/layers/memory/gpu/mod.rs

use crate::compute_manager::gpu::compute::GpuCompute;
use crate::compute_manager::matrix_buffer::MatrixBufferHandle;

impl GpuCompute {
    pub fn run_memory_forward_buffered_handle(
        &self,
        input: &MatrixBufferHandle,
        output: &MatrixBufferHandle,
        alpha: f32,
        memory_idx: usize,
    ) {
        assert!(input.is_gpu() && output.is_gpu(), "Handles must be GPU");
        let batch = input.rows();
        let features = input.cols();
        let total = batch * features;
        assert_eq!(output.rows(), batch);
        assert_eq!(output.cols(), features);

        let in_buf = self.get_gpu_subbuffer_from_handle(input);
        let out_buf = self.get_gpu_subbuffer_from_handle(output);

        let state = {
            let mut states = self.memory_states.lock().unwrap();
            if let Some((buf, _)) = states.get(&memory_idx) {
                buf.clone()
            } else {
                let input_vec = self.download_gpu_handle_to_vec(input);
                let mut state_vec = Vec::with_capacity(2 * features);
                for c in 0..features {
                    let first_val = input_vec[c * batch];
                    state_vec.push(first_val);
                }
                for c in 0..features {
                    let first_val = input_vec[c * batch];
                    state_vec.push(first_val);
                }
                let (state_buf, raw_id) = self.upload_to_temp_buffer(&state_vec);
                states.insert(memory_idx, (state_buf.clone(), raw_id));
                state_buf
            }
        };

        let pipeline = self.pipeline_cache.memory_fwd.clone();
        let push = [batch as u32, features as u32, alpha.to_bits()];
        self.run_compute_shader(
            pipeline,
            &[(0, in_buf), (1, state), (2, out_buf)],
            &push,
            total,
        );
    }

    pub fn run_memory_backward_buffered_handle(
        &self,
        grad_out: &MatrixBufferHandle,
        grad_input: &MatrixBufferHandle,
        alpha: f32,
    ) {
        assert!(grad_out.is_gpu() && grad_input.is_gpu(), "Handles must be GPU");
        let rows = grad_out.rows();
        let cols = grad_out.cols();
        let total = rows * cols;
        assert_eq!(grad_input.rows(), rows);
        assert_eq!(grad_input.cols(), cols);

        let go_buf = self.get_gpu_subbuffer_from_handle(grad_out);
        let gi_buf = self.get_gpu_subbuffer_from_handle(grad_input);

        let pipeline = self.pipeline_cache.memory_bwd.clone();
        let push = [total as u32, alpha.to_bits()];
        self.run_compute_shader(
            pipeline,
            &[(0, go_buf), (1, gi_buf)],
            &push,
            total,
        );
    }
}