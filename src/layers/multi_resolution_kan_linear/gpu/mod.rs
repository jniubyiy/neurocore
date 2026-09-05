// src/layers/multi_resolution_kan_linear/gpu/mod.rs

pub mod pipeline;

use crate::compute_manager::gpu::compute::GpuCompute;
use crate::compute_manager::matrix_buffer::view::MatrixBufferView;
use crate::compute_manager::matrix_buffer::MatrixBufferHandle;
use vulkano::buffer::Subbuffer;

/// Вспомогательная функция: получает `Subbuffer<[f32]>` из `MatrixBufferView`.
fn subbuffer_from_view(gpu: &GpuCompute, view: &MatrixBufferView) -> Subbuffer<[f32]> {
    let parent_sub = gpu.get_gpu_subbuffer_from_handle(view.parent_handle());
    let start = view.offset_elements() as u64;
    let end = (view.offset_elements() + view.len()) as u64;
    parent_sub.slice(start..end)
}

impl GpuCompute {
    /// Прямой проход MultiResolutionKANLinear на GPU.
    ///
    /// Параметры (coarse, fine, bias) передаются как `MatrixBufferView`,
    /// ссылающийся на часть общего GPU-буфера параметров сегмента.
    /// Вход и выход — GPU-дескрипторы.
    /// Внутри используется один compute-шейдер, который вычисляет
    /// взвешенную сумму грубой и точной интерполяций.
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

        // Проверка длины параметров: in*out*(4+8) + out
        let expected_param_len = in_features * out_features * 12 + out_features;
        assert_eq!(params.len(), expected_param_len, "Params length mismatch");

        let in_buf = self.get_gpu_subbuffer_from_handle(input);
        let params_buf = subbuffer_from_view(self, params);
        let out_buf = self.get_gpu_subbuffer_from_handle(output);

        let pipeline = &self.multi_resolution_kan_linear_pipelines().forward;
        let push = [batch as u32, in_features as u32, out_features as u32];

        self.run_compute_shader(
            pipeline,
            &[(0, in_buf), (1, params_buf), (2, out_buf)],
            &push,
            batch * out_features,
        );
    }

    /// Обратный проход MultiResolutionKANLinear на GPU.
    ///
    /// Градиенты по параметрам записываются в `grad_params` (часть общего GPU-буфера
    /// градиентов). Вход/выходные градиенты — GPU-дескрипторы.
    /// Перед вызовом область `grad_params` обнуляется, так как шейдер использует
    /// атомарное накопление. Затем запускаются две фазы:
    /// фаза 0 — накопление градиентов параметров,
    /// фаза 1 — вычисление градиента по входу.
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

        let expected_param_len = in_features * out_features * 12 + out_features;
        assert_eq!(params.len(), expected_param_len, "Params length mismatch");
        assert_eq!(grad_params.len(), expected_param_len, "grad_params length mismatch");

        // Обнуляем градиенты параметров
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
        let gi_buf = self.get_gpu_subbuffer_from_handle(grad_input);
        let grad_params_buf = subbuffer_from_view(self, grad_params);

        let pipeline = &self.multi_resolution_kan_linear_pipelines().backward;

        // Фаза 0: градиенты параметров
        {
            let push = [
                batch as u32,
                in_features as u32,
                out_features as u32,
                0u32, // phase = 0
            ];
            self.run_compute_shader(
                pipeline,
                &[
                    (0, in_buf.clone()),
                    (1, go_buf.clone()),
                    (2, params_buf.clone()),
                    (3, grad_params_buf.clone()),
                    (4, gi_buf.clone()),
                ],
                &push,
                batch * out_features,
            );
        }

        // Фаза 1: градиент по входу
        {
            let push = [
                batch as u32,
                in_features as u32,
                out_features as u32,
                1u32, // phase = 1
            ];
            self.run_compute_shader(
                pipeline,
                &[
                    (0, in_buf.clone()),
                    (1, go_buf.clone()),
                    (2, params_buf.clone()),
                    (3, grad_params_buf.clone()),
                    (4, gi_buf.clone()),
                ],
                &push,
                batch * in_features,
            );
        }
    }
}