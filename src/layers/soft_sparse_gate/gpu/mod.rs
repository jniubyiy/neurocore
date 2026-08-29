// src/layers/soft_sparse_gate/gpu/mod.rs

pub mod pipeline;

use crate::compute_manager::gpu::compute::GpuCompute;
use crate::compute_manager::matrix_buffer::view::MatrixBufferView;
use crate::compute_manager::matrix_buffer::MatrixBufferHandle;
use vulkano::buffer::Subbuffer;

/// Вспомогательная функция: получает `Subbuffer<[f32]>` из `MatrixBufferView`,
/// используя смещение и длину. Родительский буфер должен быть GPU.
fn subbuffer_from_view(gpu: &GpuCompute, view: &MatrixBufferView) -> Subbuffer<[f32]> {
    let parent_sub = gpu.get_gpu_subbuffer_from_handle(view.parent_handle());
    let start = view.offset_elements() as u64;
    let end = (view.offset_elements() + view.len()) as u64;
    parent_sub.slice(start..end)
}

impl GpuCompute {
    /// Прямой проход SoftSparseGate на GPU.
    ///
    /// Пороги передаются как `MatrixBufferView`, ссылающийся на часть
    /// общего GPU-буфера параметров сегмента. Вход и выход — GPU-дескрипторы.
    pub fn run_softsparse_forward_buffered_handle(
        &self,
        input: &MatrixBufferHandle,
        thresholds: &MatrixBufferView,
        temperature: f32,
        output: &MatrixBufferHandle,
    ) {
        assert!(input.is_gpu(), "Input handle must be GPU");
        assert!(output.is_gpu(), "Output handle must be GPU");
        assert!(thresholds.is_gpu(), "Thresholds view must point to GPU buffer");

        let batch = input.rows();
        let features = input.cols();
        let total = batch * features;
        assert_eq!(output.rows(), batch);
        assert_eq!(output.cols(), features);
        assert_eq!(thresholds.len(), features, "Thresholds length must equal features");

        let in_buf = self.get_gpu_subbuffer_from_handle(input);
        let thresh_buf = subbuffer_from_view(self, thresholds);
        let out_buf = self.get_gpu_subbuffer_from_handle(output);

        let pipeline = self.soft_sparse_gate_pipelines().forward.clone();
        let push = [total as u32, temperature.to_bits(), features as u32];
        self.run_compute_shader(
            pipeline,
            &[(0, in_buf), (1, thresh_buf), (2, out_buf)],
            &push,
            total,
        );
    }

    /// Обратный проход SoftSparseGate на GPU.
    ///
    /// Градиенты порогов записываются непосредственно в `grad_thresh`
    /// (часть общего GPU-буфера градиентов). Вход/выходные градиенты — GPU-дескрипторы.
    pub fn run_softsparse_backward_buffered_handle(
        &self,
        input: &MatrixBufferHandle,
        grad_out: &MatrixBufferHandle,
        thresholds: &MatrixBufferView,
        temperature: f32,
        grad_input: &MatrixBufferHandle,
        grad_thresh: &MatrixBufferView,
    ) {
        assert!(input.is_gpu(), "Input handle must be GPU");
        assert!(grad_out.is_gpu(), "grad_out handle must be GPU");
        assert!(grad_input.is_gpu(), "grad_input handle must be GPU");
        assert!(grad_thresh.is_gpu(), "grad_thresh view must point to GPU buffer");
        assert!(thresholds.is_gpu(), "Thresholds view must point to GPU buffer");

        let batch = input.rows();
        let features = input.cols();
        let total = batch * features;
        assert_eq!(grad_out.rows(), batch);
        assert_eq!(grad_out.cols(), features);
        assert_eq!(grad_input.rows(), batch);
        assert_eq!(grad_input.cols(), features);
        assert_eq!(grad_thresh.len(), features, "grad_thresh length must equal features");
        assert_eq!(thresholds.len(), features, "Thresholds length must equal features");

        let in_buf = self.get_gpu_subbuffer_from_handle(input);
        let go_buf = self.get_gpu_subbuffer_from_handle(grad_out);
        let thresh_buf = subbuffer_from_view(self, thresholds);
        let gi_buf = self.get_gpu_subbuffer_from_handle(grad_input);
        let gthresh_buf = subbuffer_from_view(self, grad_thresh);

        let pipeline = self.soft_sparse_gate_pipelines().backward.clone();
        let push = [total as u32, temperature.to_bits(), features as u32];
        self.run_compute_shader(
            pipeline,
            &[
                (0, in_buf),
                (1, go_buf),
                (2, thresh_buf),
                (3, gi_buf),
                (4, gthresh_buf),
            ],
            &push,
            total,
        );
    }
}