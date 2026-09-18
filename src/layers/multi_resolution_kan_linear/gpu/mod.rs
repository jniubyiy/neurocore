// src/layers/multi_resolution_kan_linear/gpu/mod.rs

pub mod pipeline;

use crate::compute_manager::operators_v2::gpu_v2::compute::GpuCompute;
use crate::compute_manager::operators_v2::memory_v2::buffer::view::MatrixBufferView;
use crate::compute_manager::operators_v2::memory_v2::buffer::MatrixBufferHandle;
use vulkano::buffer::Subbuffer;

fn subbuffer_from_view(gpu: &GpuCompute, view: &MatrixBufferView) -> Subbuffer<[f32]> {
    let parent_sub = gpu.get_gpu_subbuffer_from_handle(view.parent_handle());
    let start = view.offset_elements() as u64;
    let end = (view.offset_elements() + view.len()) as u64;
    parent_sub.slice(start..end)
}

impl GpuCompute {
    /// Прямой проход MultiResolutionKANLinear (v2) на GPU.
    ///
    /// Две фазы:
    ///   1. fwd_edge  → временный буфер edge_out[batch · out · in]
    ///   2. fwd_reduce → y[batch · out]
    pub fn run_multi_resolution_kan_linear_forward_buffered_handle(
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
        assert_eq!(output.rows(), batch, "Output rows mismatch");

        let expected_param_len = in_features * out_features * 21 + out_features;
        assert_eq!(params.len(), expected_param_len, "Params length mismatch");

        let edge_elems = batch * out_features * in_features;

        let in_buf = self.get_gpu_subbuffer_from_handle(input);
        let params_buf = subbuffer_from_view(self, params);
        let out_buf = self.get_gpu_subbuffer_from_handle(output);

        // 1. Фаза 1: edge-вклады.
        let (edge_buf, edge_raw) = self.acquire_temp_buffer(edge_elems);
        let pipelines = self.multi_resolution_kan_linear_pipelines();
        let push = [batch as u32, in_features as u32, out_features as u32];

        self.run_compute_shader(
            &pipelines.fwd_edge,
            &[
                (0, in_buf.clone()),
                (1, params_buf.clone()),
                (2, edge_buf.clone()),
            ],
            &push,
            edge_elems,
        );

        // 2. Фаза 2: reduce по i + bias.
        self.run_compute_shader(
            &pipelines.fwd_reduce,
            &[
                (0, edge_buf.clone()),
                (1, params_buf),
                (2, out_buf),
            ],
            &push,
            batch * out_features,
        );

        self.release_temp_buffer(edge_buf, edge_raw);
    }

    /// Обратный проход MultiResolutionKANLinear (v2) на GPU.
    ///
    /// Две фазы:
    ///   1. bwd_edge → атомарное накопление в grad_params
    ///   2. bwd_gi   → grad_input
    ///
    /// Перед вызовом grad_params должен быть обнулён (это делает processor.rs).
    pub fn run_multi_resolution_kan_linear_backward_buffered_handle(
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
        assert_eq!(grad_out.rows(), batch, "grad_out rows mismatch");
        assert_eq!(grad_input.rows(), batch, "grad_input rows mismatch");
        assert_eq!(grad_input.cols(), in_features, "grad_input cols mismatch");

        let expected_param_len = in_features * out_features * 21 + out_features;
        assert_eq!(params.len(), expected_param_len, "Params length mismatch");
        assert_eq!(grad_params.len(), expected_param_len, "grad_params length mismatch");

        let in_buf = self.get_gpu_subbuffer_from_handle(input);
        let go_buf = self.get_gpu_subbuffer_from_handle(grad_out);
        let params_buf = subbuffer_from_view(self, params);
        let gi_buf = self.get_gpu_subbuffer_from_handle(grad_input);
        let grad_params_buf = subbuffer_from_view(self, grad_params);

        let pipelines = self.multi_resolution_kan_linear_pipelines();
        let push = [batch as u32, in_features as u32, out_features as u32];

        // 1. Градиенты параметров (атомарно).
        self.run_compute_shader(
            &pipelines.bwd_edge,
            &[
                (0, in_buf.clone()),
                (1, go_buf.clone()),
                (2, params_buf.clone()),
                (3, grad_params_buf),
            ],
            &push,
            batch * out_features * in_features,
        );

        // 2. Градиент по входу.
        self.run_compute_shader(
            &pipelines.bwd_gi,
            &[
                (0, in_buf),
                (1, go_buf),
                (2, params_buf),
                (3, gi_buf),
            ],
            &push,
            batch * in_features,
        );
    }
}