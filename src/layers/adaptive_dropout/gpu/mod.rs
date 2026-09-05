// src/layers/adaptive_dropout/gpu/mod.rs

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
    /// Прямой проход AdaptiveDropout на GPU.
    ///
    /// `params_view` — представление буфера параметров, содержащего `theta` и `T`
    /// (всего 2*features элементов: сначала `theta`, затем `T`).
    /// `mask_out` и `arg_out` — GPU-буферы размера `batch * features`,
    /// в которые записываются бинарная маска `z` и аргумент `a = (|x| - theta)/T`
    /// для последующего обратного прохода.
    /// `seed` — 32-битное зерно для генератора псевдослучайных чисел на GPU.
    pub fn run_adaptive_dropout_forward_buffered_handle(
        &self,
        input: &MatrixBufferHandle,
        params: &MatrixBufferView,
        output: &MatrixBufferHandle,
        mask_out: &MatrixBufferHandle,
        arg_out: &MatrixBufferHandle,
        seed: u32,
    ) {
        assert!(input.is_gpu(), "Input handle must be GPU");
        assert!(output.is_gpu(), "Output handle must be GPU");
        assert!(mask_out.is_gpu(), "mask_out handle must be GPU");
        assert!(arg_out.is_gpu(), "arg_out handle must be GPU");
        assert!(params.is_gpu(), "Params view must point to GPU buffer");

        let batch = input.rows();
        let features = input.cols();
        let total = batch * features;
        assert_eq!(output.rows(), batch);
        assert_eq!(output.cols(), features);
        assert_eq!(mask_out.rows() * mask_out.cols(), total, "mask_out size mismatch");
        assert_eq!(arg_out.rows() * arg_out.cols(), total, "arg_out size mismatch");
        assert_eq!(params.len(), 2 * features, "Params length must be 2*features");

        let in_buf = self.get_gpu_subbuffer_from_handle(input);
        let params_buf = subbuffer_from_view(self, params);
        let out_buf = self.get_gpu_subbuffer_from_handle(output);
        let mask_buf = self.get_gpu_subbuffer_from_handle(mask_out);
        let arg_buf = self.get_gpu_subbuffer_from_handle(arg_out);

        let pipeline = &self.adaptive_dropout_pipelines().forward;
        let push = [total as u32, features as u32, seed];

        self.run_compute_shader(
            pipeline,
            &[
                (0, in_buf),
                (1, params_buf),
                (2, out_buf),
                (3, mask_buf),
                (4, arg_buf),
            ],
            &push,
            total,
        );
    }

    /// Обратный проход AdaptiveDropout на GPU.
    ///
    /// `mask` и `arg` — GPU-буферы, сохранённые при прямом проходе.
    /// `params_view` — тот же view на параметры (theta, T).
    /// `grad_params_view` — представление градиентов по параметрам (2*features).
    /// Перед вызовом область `grad_params_view` обнуляется, так как шейдер
    /// использует атомарное накопление.
    pub fn run_adaptive_dropout_backward_buffered_handle(
        &self,
        input: &MatrixBufferHandle,
        grad_out: &MatrixBufferHandle,
        params: &MatrixBufferView,
        mask: &MatrixBufferHandle,
        arg: &MatrixBufferHandle,
        grad_input: &MatrixBufferHandle,
        grad_params: &MatrixBufferView,
    ) {
        assert!(input.is_gpu(), "Input handle must be GPU");
        assert!(grad_out.is_gpu(), "grad_out handle must be GPU");
        assert!(grad_input.is_gpu(), "grad_input handle must be GPU");
        assert!(mask.is_gpu(), "mask handle must be GPU");
        assert!(arg.is_gpu(), "arg handle must be GPU");
        assert!(params.is_gpu(), "Params view must point to GPU buffer");
        assert!(grad_params.is_gpu(), "grad_params view must point to GPU buffer");

        let batch = input.rows();
        let features = input.cols();
        let total = batch * features;
        assert_eq!(grad_out.rows(), batch);
        assert_eq!(grad_out.cols(), features);
        assert_eq!(grad_input.rows(), batch);
        assert_eq!(grad_input.cols(), features);
        assert_eq!(mask.rows() * mask.cols(), total, "mask size mismatch");
        assert_eq!(arg.rows() * arg.cols(), total, "arg size mismatch");
        assert_eq!(params.len(), 2 * features, "Params length mismatch");
        assert_eq!(grad_params.len(), 2 * features, "grad_params length mismatch");

        // Обнуляем градиенты по параметрам перед накоплением
        let zero_handle = self.upload_vec_to_gpu_handle(
            &vec![0.0f32; grad_params.len()],
            grad_params.len(),
            1,
        );
        self.copy_gpu_handle_region(
            &zero_handle,
            grad_params.parent_handle(),
            0,
            grad_params.offset_elements(),
            grad_params.len(),
        );

        let in_buf = self.get_gpu_subbuffer_from_handle(input);
        let go_buf = self.get_gpu_subbuffer_from_handle(grad_out);
        let params_buf = subbuffer_from_view(self, params);
        let mask_buf = self.get_gpu_subbuffer_from_handle(mask);
        let arg_buf = self.get_gpu_subbuffer_from_handle(arg);
        let gi_buf = self.get_gpu_subbuffer_from_handle(grad_input);
        let grad_params_buf = subbuffer_from_view(self, grad_params);

        let pipeline = &self.adaptive_dropout_pipelines().backward;
        let push = [total as u32, features as u32];

        self.run_compute_shader(
            pipeline,
            &[
                (0, in_buf),
                (1, go_buf),
                (2, params_buf),
                (3, mask_buf),
                (4, arg_buf),
                (5, gi_buf),
                (6, grad_params_buf),
            ],
            &push,
            total,
        );
    }
}