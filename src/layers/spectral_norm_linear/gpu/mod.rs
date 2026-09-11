// src/layers/spectral_norm_linear/gpu/mod.rs

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

const POWER_EPS: f32 = 1e-12;

impl GpuCompute {
    /// Прямой проход SpectrallyNormalizedLinear на GPU.
    ///
    /// Power iteration (поэтапно, без локальных массивов, без MAX_DIM):
    ///   matvec_v → normalize(v) → matvec_u → normalize(u) → sigma
    /// Затем основной forward с эффективным масштабом scale / sigma.
    pub fn run_spectral_norm_linear_forward_buffered_handle(
        &self,
        input: &MatrixBufferHandle,
        params: &MatrixBufferView,
        scale: f32,
        output: &MatrixBufferHandle,
        u_state: &MatrixBufferHandle,
        v_state: &MatrixBufferHandle,
        sigma_state: &MatrixBufferHandle,
    ) {
        assert!(input.is_gpu(), "Input handle must be GPU");
        assert!(output.is_gpu(), "Output handle must be GPU");
        assert!(params.is_gpu(), "Params view must point to GPU buffer");
        assert!(u_state.is_gpu(), "u_state handle must be GPU");
        assert!(v_state.is_gpu(), "v_state handle must be GPU");
        assert!(sigma_state.is_gpu(), "sigma_state handle must be GPU");

        let batch = input.rows();
        let in_feat = input.cols();
        let out_feat = output.cols();
        assert_eq!(output.rows(), batch, "Output rows mismatch");
        assert_eq!(
            params.len(),
            out_feat * in_feat + out_feat + 1,
            "Params length mismatch"
        );
        assert_eq!(u_state.rows() * u_state.cols(), in_feat, "u_state size mismatch");
        assert_eq!(v_state.rows() * v_state.cols(), out_feat, "v_state size mismatch");
        assert_eq!(sigma_state.rows() * sigma_state.cols(), 1, "sigma_state size mismatch");

        // View на W (row-major) и bias.
        let w_view = MatrixBufferView::with_shape(
            params.parent_handle().clone(),
            params.offset_elements(),
            out_feat * in_feat,
            out_feat,
            in_feat,
        );
        let b_start = out_feat * in_feat;
        let b_view = MatrixBufferView::new(
            params.parent_handle().clone(),
            params.offset_elements() + b_start,
            out_feat,
        );

        let in_buf = self.get_gpu_subbuffer_from_handle(input);
        let out_buf = self.get_gpu_subbuffer_from_handle(output);
        let w_buf = subbuffer_from_view(self, &w_view);
        let b_buf = subbuffer_from_view(self, &b_view);
        let u_buf = self.get_gpu_subbuffer_from_handle(u_state);
        let v_buf = self.get_gpu_subbuffer_from_handle(v_state);
        let sigma_buf = self.get_gpu_subbuffer_from_handle(sigma_state);

        let pipelines = self.spectral_norm_linear_pipelines();

        let push_io = [in_feat as u32, out_feat as u32];

        // 1. v = W · u.
        self.run_compute_shader(
            &pipelines.matvec_v,
            &[
                (0, w_buf.clone()),
                (1, u_buf.clone()),
                (2, v_buf.clone()),
            ],
            &push_io,
            out_feat,
        );

        // 2. Нормализация v.
        let push_norm_v = [out_feat as u32, POWER_EPS.to_bits()];
        self.run_compute_shader_with_dispatch(
            &pipelines.normalize,
            &[(0, v_buf.clone())],
            &push_norm_v,
            [1, 1, 1],
        );

        // 3. u = Wᵀ · v.
        self.run_compute_shader(
            &pipelines.matvec_u,
            &[
                (0, w_buf.clone()),
                (1, v_buf.clone()),
                (2, u_buf.clone()),
            ],
            &push_io,
            in_feat,
        );

        // 4. Нормализация u.
        let push_norm_u = [in_feat as u32, POWER_EPS.to_bits()];
        self.run_compute_shader_with_dispatch(
            &pipelines.normalize,
            &[(0, u_buf.clone())],
            &push_norm_u,
            [1, 1, 1],
        );

        // 5. sigma = uᵀ W v.
        self.run_compute_shader_with_dispatch(
            &pipelines.sigma,
            &[
                (0, w_buf.clone()),
                (1, u_buf.clone()),
                (2, v_buf.clone()),
                (3, sigma_buf.clone()),
            ],
            &push_io,
            [1, 1, 1],
        );

        // 6. Скачиваем sigma для push-константы forward.
        let sigma = {
            let sigma_vec = self.download_gpu_handle_to_vec(sigma_state);
            sigma_vec[0]
        };

        // 7. Основной forward.
        let push_fwd = [
            batch as u32,
            in_feat as u32,
            out_feat as u32,
            scale.to_bits(),
            sigma.to_bits(),
        ];
        self.run_compute_shader(
            &pipelines.forward,
            &[
                (0, in_buf),
                (1, w_buf),
                (2, b_buf),
                (3, out_buf),
            ],
            &push_fwd,
            batch * out_feat,
        );
    }

    /// Обратный проход SpectrallyNormalizedLinear на GPU.
    ///
    /// Не менялся: единый шейдер с phase 0/1.
    pub fn run_spectral_norm_linear_backward_buffered_handle(
        &self,
        input: &MatrixBufferHandle,
        grad_out: &MatrixBufferHandle,
        params: &MatrixBufferView,
        grad_input: &MatrixBufferHandle,
        grad_params: &MatrixBufferView,
        scale: f32,
        sigma: f32,
    ) {
        assert!(input.is_gpu(), "Input handle must be GPU");
        assert!(grad_out.is_gpu(), "grad_out handle must be GPU");
        assert!(grad_input.is_gpu(), "grad_input handle must be GPU");
        assert!(grad_params.is_gpu(), "grad_params view must point to GPU buffer");
        assert!(params.is_gpu(), "Params view must point to GPU buffer");

        let batch = input.rows();
        let in_feat = input.cols();
        let out_feat = grad_out.cols();
        assert_eq!(grad_out.rows(), batch, "grad_out rows mismatch");
        assert_eq!(grad_input.rows(), batch, "grad_input rows mismatch");
        assert_eq!(grad_input.cols(), in_feat, "grad_input cols mismatch");
        assert_eq!(
            params.len(),
            out_feat * in_feat + out_feat + 1,
            "Params length mismatch"
        );
        assert_eq!(
            grad_params.len(),
            out_feat * in_feat + out_feat + 1,
            "Grad params length mismatch"
        );

        // Обнуляем grad_params.
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

        let w_start = 0;
        let b_start = out_feat * in_feat;
        let scale_idx = b_start + out_feat;

        let w_view = MatrixBufferView::with_shape(
            params.parent_handle().clone(),
            params.offset_elements() + w_start,
            out_feat * in_feat,
            out_feat,
            in_feat,
        );
        let gw_view = MatrixBufferView::with_shape(
            grad_params.parent_handle().clone(),
            grad_params.offset_elements() + w_start,
            out_feat * in_feat,
            out_feat,
            in_feat,
        );
        let gb_view = MatrixBufferView::new(
            grad_params.parent_handle().clone(),
            grad_params.offset_elements() + b_start,
            out_feat,
        );
        let gscale_view = MatrixBufferView::new(
            grad_params.parent_handle().clone(),
            grad_params.offset_elements() + scale_idx,
            1,
        );

        let in_buf = self.get_gpu_subbuffer_from_handle(input);
        let go_buf = self.get_gpu_subbuffer_from_handle(grad_out);
        let w_buf = subbuffer_from_view(self, &w_view);
        let gw_buf = subbuffer_from_view(self, &gw_view);
        let gb_buf = subbuffer_from_view(self, &gb_view);
        let gscale_buf = subbuffer_from_view(self, &gscale_view);
        let gi_buf = self.get_gpu_subbuffer_from_handle(grad_input);

        let pipeline = &self.spectral_norm_linear_pipelines().backward;

        // Фаза 0: градиенты параметров.
        let push_bwd_params = [
            batch as u32,
            in_feat as u32,
            out_feat as u32,
            scale.to_bits(),
            sigma.to_bits(),
            0u32,
        ];
        self.run_compute_shader(
            pipeline,
            &[
                (0, in_buf.clone()),
                (1, go_buf.clone()),
                (2, w_buf.clone()),
                (3, gw_buf.clone()),
                (4, gb_buf.clone()),
                (5, gscale_buf.clone()),
                (6, gi_buf.clone()),
            ],
            &push_bwd_params,
            batch * out_feat,
        );

        // Фаза 1: градиент по входу.
        let push_bwd_input = [
            batch as u32,
            in_feat as u32,
            out_feat as u32,
            scale.to_bits(),
            sigma.to_bits(),
            1u32,
        ];
        self.run_compute_shader(
            pipeline,
            &[
                (0, in_buf.clone()),
                (1, go_buf.clone()),
                (2, w_buf.clone()),
                (3, gw_buf.clone()),
                (4, gb_buf.clone()),
                (5, gscale_buf.clone()),
                (6, gi_buf.clone()),
            ],
            &push_bwd_input,
            batch * in_feat,
        );
    }
}