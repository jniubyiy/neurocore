// src/layers/concrete_dropout/gpu/mod.rs

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
    /// Прямой проход ConcreteDropout на GPU.
    ///
    /// `logit_p_view` — представление единственного обучаемого параметра `logit_p`
    /// (размер 1) в общем GPU-буфере параметров.
    /// `arg_out` — GPU-буфер размера `batch * features`, в который записываются
    /// аргументы сигмоиды `a = (logit_p + log(u) - log(1-u)) / temperature`
    /// для последующего обратного прохода.
    /// `seed` — 32-битное зерно для генератора псевдослучайных чисел на GPU.
    pub fn run_concrete_dropout_forward_buffered_handle(
        &self,
        input: &MatrixBufferHandle,
        logit_p_view: &MatrixBufferView,
        temperature: f32,
        output: &MatrixBufferHandle,
        arg_out: &MatrixBufferHandle,
        seed: u32,
    ) {
        assert!(input.is_gpu(), "Input handle must be GPU");
        assert!(output.is_gpu(), "Output handle must be GPU");
        assert!(arg_out.is_gpu(), "arg_out handle must be GPU");
        assert!(logit_p_view.is_gpu(), "logit_p view must point to GPU buffer");
        assert_eq!(logit_p_view.len(), 1, "logit_p length must be 1");

        let batch = input.rows();
        let features = input.cols();
        let total = batch * features;
        assert_eq!(output.rows(), batch);
        assert_eq!(output.cols(), features);
        assert_eq!(arg_out.rows() * arg_out.cols(), total, "arg_out size mismatch");

        let in_buf = self.get_gpu_subbuffer_from_handle(input);
        let logit_buf = subbuffer_from_view(self, logit_p_view);
        let out_buf = self.get_gpu_subbuffer_from_handle(output);
        let arg_buf = self.get_gpu_subbuffer_from_handle(arg_out);

        let pipeline = &self.concrete_dropout_pipelines().forward;
        let push = [total as u32, temperature.to_bits(), seed];

        self.run_compute_shader(
            pipeline,
            &[
                (0, in_buf),
                (1, logit_buf),
                (2, out_buf),
                (3, arg_buf),
            ],
            &push,
            total,
        );
    }

    /// Обратный проход ConcreteDropout на GPU.
    ///
    /// `arg_in` — GPU-буфер, содержащий аргументы `a`, сохранённые при прямом проходе.
    /// `grad_logit_p_view` — представление градиента по параметру `logit_p`
    /// (размер 1) в общем GPU-буфере градиентов.
    /// Перед вызовом область `grad_logit_p_view` обнуляется, так как шейдер
    /// использует атомарное накопление.
    pub fn run_concrete_dropout_backward_buffered_handle(
        &self,
        input: &MatrixBufferHandle,
        grad_out: &MatrixBufferHandle,
        logit_p_view: &MatrixBufferView,
        temperature: f32,
        arg_in: &MatrixBufferHandle,
        grad_input: &MatrixBufferHandle,
        grad_logit_p_view: &MatrixBufferView,
    ) {
        assert!(input.is_gpu(), "Input handle must be GPU");
        assert!(grad_out.is_gpu(), "grad_out handle must be GPU");
        assert!(grad_input.is_gpu(), "grad_input handle must be GPU");
        assert!(arg_in.is_gpu(), "arg_in handle must be GPU");
        assert!(logit_p_view.is_gpu(), "logit_p view must point to GPU buffer");
        assert!(grad_logit_p_view.is_gpu(), "grad_logit_p view must point to GPU buffer");
        assert_eq!(logit_p_view.len(), 1, "logit_p length must be 1");
        assert_eq!(grad_logit_p_view.len(), 1, "grad_logit_p length must be 1");

        let batch = input.rows();
        let features = input.cols();
        let total = batch * features;
        assert_eq!(grad_out.rows(), batch);
        assert_eq!(grad_out.cols(), features);
        assert_eq!(grad_input.rows(), batch);
        assert_eq!(grad_input.cols(), features);
        assert_eq!(arg_in.rows() * arg_in.cols(), total, "arg_in size mismatch");

        // Обнуляем градиент по logit_p
        let zero_handle = self.upload_vec_to_gpu_handle(
            &[0.0f32],
            1,
            1,
        );
        self.copy_gpu_handle_region(
            &zero_handle,
            grad_logit_p_view.parent_handle(),
            0,
            grad_logit_p_view.offset_elements(),
            1,
        );

        let in_buf = self.get_gpu_subbuffer_from_handle(input);
        let go_buf = self.get_gpu_subbuffer_from_handle(grad_out);
        let logit_buf = subbuffer_from_view(self, logit_p_view);
        let arg_buf = self.get_gpu_subbuffer_from_handle(arg_in);
        let gi_buf = self.get_gpu_subbuffer_from_handle(grad_input);
        let grad_logit_buf = subbuffer_from_view(self, grad_logit_p_view);

        let pipeline = &self.concrete_dropout_pipelines().backward;
        let push = [total as u32, temperature.to_bits()];

        self.run_compute_shader(
            pipeline,
            &[
                (0, in_buf),
                (1, go_buf),
                (2, logit_buf),
                (3, arg_buf),
                (4, gi_buf),
                (5, grad_logit_buf),
            ],
            &push,
            total,
        );
    }
}