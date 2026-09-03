// src/layers/adaptive_activation/gpu/mod.rs

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
    /// Прямой проход AdaptivePerFeatureActivation на GPU.
    ///
    /// Параметры (логиты) передаются как `MatrixBufferView`, ссылающийся на часть
    /// общего GPU-буфера параметров сегмента. Вход и выход — GPU-дескрипторы.
    pub fn run_adaptive_activation_forward_buffered_handle(
        &self,
        input: &MatrixBufferHandle,
        params: &MatrixBufferView,
        num_activations: usize,
        output: &MatrixBufferHandle,
    ) {
        assert!(input.is_gpu(), "Input handle must be GPU");
        assert!(output.is_gpu(), "Output handle must be GPU");
        assert!(params.is_gpu(), "Params view must point to GPU buffer");

        let batch = input.rows();
        let features = input.cols();
        let total = batch * features;
        assert_eq!(output.rows(), batch);
        assert_eq!(output.cols(), features);
        assert_eq!(params.len(), num_activations * features, "Params length mismatch");

        let in_buf = self.get_gpu_subbuffer_from_handle(input);
        let out_buf = self.get_gpu_subbuffer_from_handle(output);
        let params_buf = subbuffer_from_view(self, params);

        let pipeline = &self.adaptive_activation_pipelines().forward;
        let push = [total as u32, features as u32, num_activations as u32];
        self.run_compute_shader(
            pipeline,
            &[(0, in_buf), (1, params_buf), (2, out_buf)],
            &push,
            total,
        );
    }

    /// Обратный проход AdaptivePerFeatureActivation на GPU.
    ///
    /// Градиенты по логитам записываются в `grad_params` (часть общего GPU-буфера
    /// градиентов). Вход/выходные градиенты — GPU-дескрипторы.
    /// Перед вызовом необходимо, чтобы область `grad_params` была заполнена нулями.
    pub fn run_adaptive_activation_backward_buffered_handle(
        &self,
        input: &MatrixBufferHandle,
        grad_output: &MatrixBufferHandle,
        params: &MatrixBufferView,
        num_activations: usize,
        grad_input: &MatrixBufferHandle,
        grad_params: &MatrixBufferView,
    ) {
        assert!(input.is_gpu(), "Input handle must be GPU");
        assert!(grad_output.is_gpu(), "grad_output handle must be GPU");
        assert!(grad_input.is_gpu(), "grad_input handle must be GPU");
        assert!(grad_params.is_gpu(), "grad_params view must point to GPU buffer");
        assert!(params.is_gpu(), "Params view must point to GPU buffer");

        let batch = input.rows();
        let features = input.cols();
        let total = batch * features;
        assert_eq!(grad_output.rows(), batch);
        assert_eq!(grad_output.cols(), features);
        assert_eq!(grad_input.rows(), batch);
        assert_eq!(grad_input.cols(), features);
        assert_eq!(params.len(), num_activations * features, "Params length mismatch");
        assert_eq!(grad_params.len(), num_activations * features, "grad_params length mismatch");

        let in_buf = self.get_gpu_subbuffer_from_handle(input);
        let go_buf = self.get_gpu_subbuffer_from_handle(grad_output);
        let params_buf = subbuffer_from_view(self, params);
        let gi_buf = self.get_gpu_subbuffer_from_handle(grad_input);
        let grad_params_buf = subbuffer_from_view(self, grad_params);

        let pipeline = &self.adaptive_activation_pipelines().backward;
        let push = [total as u32, features as u32, num_activations as u32];
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