// src/layers/dual_anchor/gpu/mod.rs

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
    /// Прямой проход DualAnchor на GPU.
    ///
    /// Минимальные и максимальные значения передаются как `MatrixBufferView`,
    /// ссылающиеся на части общего GPU-буфера параметров сегмента.
    /// Коэффициент `alpha` передаётся как отдельный `MatrixBufferView` длины 1.
    /// Вход и выход — GPU-дескрипторы.
    pub fn run_dualanchor_forward_buffered_handle(
        &self,
        input: &MatrixBufferHandle,
        min_vals: &MatrixBufferView,
        max_vals: &MatrixBufferView,
        alpha: &MatrixBufferView,
        output: &MatrixBufferHandle,
    ) {
        assert!(input.is_gpu(), "Input handle must be GPU");
        assert!(output.is_gpu(), "Output handle must be GPU");
        assert!(min_vals.is_gpu(), "min_vals view must point to GPU buffer");
        assert!(max_vals.is_gpu(), "max_vals view must point to GPU buffer");
        assert!(alpha.is_gpu(), "alpha view must point to GPU buffer");

        let batch = input.rows();
        let features = input.cols();
        let total = batch * features;
        assert_eq!(output.rows(), batch);
        assert_eq!(output.cols(), features);
        assert_eq!(min_vals.len(), features, "min_vals length must equal features");
        assert_eq!(max_vals.len(), features, "max_vals length must equal features");
        assert_eq!(alpha.len(), 1, "alpha length must be 1");

        let in_buf = self.get_gpu_subbuffer_from_handle(input);
        let min_buf = subbuffer_from_view(self, min_vals);
        let max_buf = subbuffer_from_view(self, max_vals);
        let alpha_buf = subbuffer_from_view(self, alpha);
        let out_buf = self.get_gpu_subbuffer_from_handle(output);

        let pipeline = self.dual_anchor_pipelines().forward.clone();
        let push = [total as u32, features as u32, 0.0f32.to_bits()];
        // ВНИМАНИЕ: alpha извлекается на CPU, так как шейдер ожидает float push constant.
        // Для простоты читаем alpha из view (CPU-side), так как длина 1.
        let alpha_val = {
            let cpu_handle = self.download_gpu_handle_to_cpu_handle(alpha.parent_handle());
            let guard = cpu_handle.read();
            let slice = guard.as_slice().unwrap();
            slice[alpha.offset_elements()]
        };
        let push = [total as u32, features as u32, alpha_val.to_bits()];

        self.run_compute_shader_with_dispatch(
            pipeline,
            &[
                (0, in_buf),
                (1, min_buf),
                (2, max_buf),
                (3, out_buf),
            ],
            &push,
            [((batch + 255) / 256) as u32, 1, 1],
        );
    }

    /// Обратный проход DualAnchor на GPU.
    ///
    /// Градиенты минимальных/максимальных значений и alpha записываются
    /// непосредственно в `grad_min`, `grad_max`, `grad_alpha` — части общего
    /// GPU-буфера градиентов. Вход/выходные градиенты — GPU-дескрипторы.
    pub fn run_dualanchor_backward_buffered_handle(
        &self,
        input: &MatrixBufferHandle,
        grad_out: &MatrixBufferHandle,
        min_vals: &MatrixBufferView,
        max_vals: &MatrixBufferView,
        alpha: &MatrixBufferView,
        grad_input: &MatrixBufferHandle,
        grad_min: &MatrixBufferView,
        grad_max: &MatrixBufferView,
        grad_alpha: &MatrixBufferView,
    ) {
        assert!(input.is_gpu(), "Input handle must be GPU");
        assert!(grad_out.is_gpu(), "grad_out handle must be GPU");
        assert!(grad_input.is_gpu(), "grad_input handle must be GPU");
        assert!(grad_min.is_gpu(), "grad_min view must point to GPU buffer");
        assert!(grad_max.is_gpu(), "grad_max view must point to GPU buffer");
        assert!(grad_alpha.is_gpu(), "grad_alpha view must point to GPU buffer");
        assert!(min_vals.is_gpu(), "min_vals view must point to GPU buffer");
        assert!(max_vals.is_gpu(), "max_vals view must point to GPU buffer");
        assert!(alpha.is_gpu(), "alpha view must point to GPU buffer");

        let batch = input.rows();
        let features = input.cols();
        let total = batch * features;
        assert_eq!(grad_out.rows(), batch);
        assert_eq!(grad_out.cols(), features);
        assert_eq!(grad_input.rows(), batch);
        assert_eq!(grad_input.cols(), features);
        assert_eq!(grad_min.len(), features, "grad_min length must equal features");
        assert_eq!(grad_max.len(), features, "grad_max length must equal features");
        assert_eq!(grad_alpha.len(), 1, "grad_alpha length must be 1");

        let in_buf = self.get_gpu_subbuffer_from_handle(input);
        let go_buf = self.get_gpu_subbuffer_from_handle(grad_out);
        let min_buf = subbuffer_from_view(self, min_vals);
        let max_buf = subbuffer_from_view(self, max_vals);
        let alpha_buf = subbuffer_from_view(self, alpha);
        let gi_buf = self.get_gpu_subbuffer_from_handle(grad_input);
        let gmin_buf = subbuffer_from_view(self, grad_min);
        let gmax_buf = subbuffer_from_view(self, grad_max);
        let galpha_buf = subbuffer_from_view(self, grad_alpha);

        // Читаем alpha из view (CPU-side)
        let alpha_val = {
            let cpu_handle = self.download_gpu_handle_to_cpu_handle(alpha.parent_handle());
            let guard = cpu_handle.read();
            let slice = guard.as_slice().unwrap();
            slice[alpha.offset_elements()]
        };

        let pipeline = self.dual_anchor_pipelines().backward.clone();
        let push = [total as u32, features as u32, alpha_val.to_bits()];

        self.run_compute_shader_with_dispatch(
            pipeline,
            &[
                (0, in_buf),
                (1, go_buf),
                (2, min_buf),
                (3, max_buf),
                (4, gi_buf),
                (5, gmin_buf),
                (6, gmax_buf),
                (7, galpha_buf),
            ],
            &push,
            [((batch + 255) / 256) as u32, 1, 1],
        );
    }
}