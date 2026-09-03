// src/layers/feature_fusion/gpu/mod.rs

pub mod pipeline;

use crate::compute_manager::gpu::compute::GpuCompute;
use crate::compute_manager::matrix_buffer::view::MatrixBufferView;
use crate::compute_manager::matrix_buffer::MatrixBufferHandle;
use vulkano::buffer::Subbuffer;

fn subbuffer_from_view(gpu: &GpuCompute, view: &MatrixBufferView) -> Subbuffer<[f32]> {
    let parent_sub = gpu.get_gpu_subbuffer_from_handle(view.parent_handle());
    let start = view.offset_elements() as u64;
    let end = (view.offset_elements() + view.len()) as u64;
    parent_sub.slice(start..end)
}

impl GpuCompute {
    pub fn run_feature_fusion_forward_buffered_handle(
        &self,
        input: &MatrixBufferHandle,
        params: &MatrixBufferView,
        output: &MatrixBufferHandle,
    ) {
        assert!(input.is_gpu(), "Input handle must be GPU");
        assert!(output.is_gpu(), "Output handle must be GPU");
        assert!(params.is_gpu(), "Params view must point to GPU buffer");

        let batch = input.rows();
        let in_features = input.cols();
        let out_features = output.cols();
        let total = batch * out_features;
        assert_eq!(output.rows(), batch);
        assert_eq!(params.len(), out_features * (in_features + 1), "Params length mismatch");

        let in_buf = self.get_gpu_subbuffer_from_handle(input);
        let params_buf = subbuffer_from_view(self, params);
        let out_buf = self.get_gpu_subbuffer_from_handle(output);

        let pipeline = &self.feature_fusion_pipelines().forward;
        let push = [batch as u32, in_features as u32, out_features as u32];
        self.run_compute_shader(
            pipeline,
            &[(0, in_buf), (1, params_buf), (2, out_buf)],
            &push,
            total,
        );
    }

    pub fn run_feature_fusion_backward_buffered_handle(
        &self,
        input: &MatrixBufferHandle,
        grad_out: &MatrixBufferHandle,
        params: &MatrixBufferView,
        grad_input: &MatrixBufferHandle,
        grad_params: &MatrixBufferView,
    ) {
        assert!(input.is_gpu(), "Input handle must be GPU");
        assert!(grad_out.is_gpu(), "grad_out handle must be GPU");
        assert!(grad_input.is_gpu(), "grad_input handle must be GPU");
        assert!(grad_params.is_gpu(), "grad_params view must point to GPU buffer");
        assert!(params.is_gpu(), "Params view must point to GPU buffer");

        let batch = input.rows();
        let in_features = input.cols();
        let out_features = grad_out.cols();
        let total = batch * in_features;
        assert_eq!(grad_out.rows(), batch);
        assert_eq!(grad_input.rows(), batch);
        assert_eq!(grad_input.cols(), in_features);
        assert_eq!(params.len(), out_features * (in_features + 1));
        assert_eq!(grad_params.len(), out_features * (in_features + 1));

        let in_buf = self.get_gpu_subbuffer_from_handle(input);
        let go_buf = self.get_gpu_subbuffer_from_handle(grad_out);
        let params_buf = subbuffer_from_view(self, params);
        let gi_buf = self.get_gpu_subbuffer_from_handle(grad_input);
        let grad_params_buf = subbuffer_from_view(self, grad_params);

        let pipeline = &self.feature_fusion_pipelines().backward;
        let push = [batch as u32, in_features as u32, out_features as u32];
        self.run_compute_shader(
            pipeline,
            &[
                (0, in_buf),
                (1, go_buf),
                (2, params_buf),
                (3, gi_buf),
                (4, grad_params_buf),
            ],
            &push,
            total,
        );
    }
}