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
    /// Прямой проход DualAnchor на GPU (column-major).
    ///
    /// Параметры `min_vals`, `max_vals`, `alpha` передаются как три отдельных
    /// `MatrixBufferView`, ссылающихся на смещения `[0, features)`,
    /// `[features, 2·features)` и `[2·features, 2·features+1)` внутри общего
    /// блока параметров. Вход и выход — GPU-дескрипторы.
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
        let out_buf = self.get_gpu_subbuffer_from_handle(output);

        // Читаем alpha на CPU — шейдер ожидает float push constant.
        let alpha_val = {
            let cpu_handle = self.download_gpu_handle_to_cpu_handle(alpha.parent_handle());
            let guard = cpu_handle.read();
            let slice = guard.as_slice().unwrap();
            slice[alpha.offset_elements()]
        };

        let pipeline = &self.dual_anchor_pipelines().forward;
        // push = [batch, features, alpha_bits] — 12 байт.
        let push = [batch as u32, features as u32, alpha_val.to_bits()];

        self.run_compute_shader(
            pipeline,
            &[
                (0, in_buf),
                (1, min_buf),
                (2, max_buf),
                (3, out_buf),
            ],
            &push,
            total,
        );
    }

    /// Обратный проход DualAnchor на GPU (column-major).
    ///
    /// Градиенты параметров записываются в `grad_params` — блок длиной
    /// `2·features + 1` (grad_min, grad_max, grad_alpha). Внутри метода
    /// создаются три отдельных Subbuffer на соответствующих смещениях.
    pub fn run_dualanchor_backward_buffered_handle(
        &self,
        input: &MatrixBufferHandle,
        grad_out: &MatrixBufferHandle,
        min_vals: &MatrixBufferView,
        max_vals: &MatrixBufferView,
        alpha: &MatrixBufferView,
        grad_input: &MatrixBufferHandle,
        grad_params: &MatrixBufferView,
    ) {
        assert!(input.is_gpu(), "Input handle must be GPU");
        assert!(grad_out.is_gpu(), "grad_out handle must be GPU");
        assert!(grad_input.is_gpu(), "grad_input handle must be GPU");
        assert!(grad_params.is_gpu(), "grad_params view must point to GPU buffer");
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

        // Единый буфер градиентов параметров: [grad_min (features), grad_max (features), grad_alpha (1)].
        assert_eq!(
            grad_params.len(),
            2 * features + 1,
            "grad_params length must be 2·features + 1"
        );

        let in_buf = self.get_gpu_subbuffer_from_handle(input);
        let go_buf = self.get_gpu_subbuffer_from_handle(grad_out);
        let min_buf = subbuffer_from_view(self, min_vals);
        let max_buf = subbuffer_from_view(self, max_vals);
        let gi_buf = self.get_gpu_subbuffer_from_handle(grad_input);

        // Разбиваем grad_params на три Subbuffer.
        let grad_params_parent = grad_params.parent_handle().clone();
        let grad_params_offset = grad_params.offset_elements();

        let gmin_view = MatrixBufferView::new(
            grad_params_parent.clone(),
            grad_params_offset,
            features,
        );
        let gmax_view = MatrixBufferView::new(
            grad_params_parent.clone(),
            grad_params_offset + features,
            features,
        );
        let galpha_view = MatrixBufferView::new(
            grad_params_parent.clone(),
            grad_params_offset + 2 * features,
            1,
        );

        let gmin_buf = subbuffer_from_view(self, &gmin_view);
        let gmax_buf = subbuffer_from_view(self, &gmax_view);
        let galpha_buf = subbuffer_from_view(self, &galpha_view);

        // Читаем alpha на CPU.
        let alpha_val = {
            let cpu_handle = self.download_gpu_handle_to_cpu_handle(alpha.parent_handle());
            let guard = cpu_handle.read();
            let slice = guard.as_slice().unwrap();
            slice[alpha.offset_elements()]
        };

        let pipeline = &self.dual_anchor_pipelines().backward;
        // push = [batch, features, alpha_bits] — 12 байт.
        let push = [batch as u32, features as u32, alpha_val.to_bits()];

        self.run_compute_shader(
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
            total,
        );
    }
}