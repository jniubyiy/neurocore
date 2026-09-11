// src/layers/splitter/gpu/mod.rs

pub mod pipeline;

use crate::compute_manager::gpu::compute::GpuCompute;
use crate::compute_manager::matrix_buffer::MatrixBufferHandle;

impl GpuCompute {
    /// Прямой проход Splitter на GPU (column-major).
    ///
    /// Один поток на выходной элемент (r, j), j ∈ [0, p+q).
    /// Dispatch = batch · (p + q).
    pub fn run_splitter_forward_buffered_handle(
        &self,
        x: &MatrixBufferHandle,
        wa: &MatrixBufferHandle,
        bias_a: &[f32],
        wb: &MatrixBufferHandle,
        bias_b: &[f32],
        out_a: &MatrixBufferHandle,
        pre_a: &MatrixBufferHandle,
        out_b: &MatrixBufferHandle,
        pre_b: &MatrixBufferHandle,
    ) {
        assert!(x.is_gpu() && wa.is_gpu() && wb.is_gpu(), "Input handles must be GPU");
        assert!(
            out_a.is_gpu() && pre_a.is_gpu() && out_b.is_gpu() && pre_b.is_gpu(),
            "Output handles must be GPU"
        );

        let batch = x.rows();
        let n = x.cols();
        let p = wa.rows();
        let q = wb.rows();
        assert_eq!(out_a.rows(), batch);
        assert_eq!(out_a.cols(), p);
        assert_eq!(pre_a.rows(), batch);
        assert_eq!(pre_a.cols(), p);
        assert_eq!(out_b.rows(), batch);
        assert_eq!(out_b.cols(), q);
        assert_eq!(pre_b.rows(), batch);
        assert_eq!(pre_b.cols(), q);

        let in_buf = self.get_gpu_subbuffer_from_handle(x);
        let wa_buf = self.get_gpu_subbuffer_from_handle(wa);
        let wb_buf = self.get_gpu_subbuffer_from_handle(wb);
        let out_a_buf = self.get_gpu_subbuffer_from_handle(out_a);
        let pre_a_buf = self.get_gpu_subbuffer_from_handle(pre_a);
        let out_b_buf = self.get_gpu_subbuffer_from_handle(out_b);
        let pre_b_buf = self.get_gpu_subbuffer_from_handle(pre_b);

        let (bias_a_buf, bias_a_raw) = self.upload_to_temp_buffer(bias_a);
        let (bias_b_buf, bias_b_raw) = self.upload_to_temp_buffer(bias_b);

        let pipeline = &self.splitter_pipelines().forward;
        let push = [batch as u32, n as u32, p as u32, q as u32];
        let total = batch * (p + q);
        self.run_compute_shader(
            pipeline,
            &[
                (0, in_buf),
                (1, wa_buf),
                (2, bias_a_buf.clone()),
                (3, wb_buf),
                (4, bias_b_buf.clone()),
                (5, out_a_buf),
                (6, pre_a_buf),
                (7, out_b_buf),
                (8, pre_b_buf),
            ],
            &push,
            total,
        );

        self.release_temp_buffer(bias_a_buf, bias_a_raw);
        self.release_temp_buffer(bias_b_buf, bias_b_raw);
    }

    /// Обратный проход Splitter на GPU (column-major).
    ///
    /// Два этапа:
    ///   1. bwd_dx      — градиент по входу x. Dispatch = batch · n.
    ///   2. bwd_params  — градиенты wa, wb, bias_a, bias_b.
    ///                    Dispatch = p·n + q·n + p + q.
    pub fn run_splitter_backward_buffered_handle(
        &self,
        x: &MatrixBufferHandle,
        da: &MatrixBufferHandle,
        db: &MatrixBufferHandle,
        pre_a: &MatrixBufferHandle,
        pre_b: &MatrixBufferHandle,
        wa: &MatrixBufferHandle,
        wb: &MatrixBufferHandle,
        dx: &MatrixBufferHandle,
        d_wa: &MatrixBufferHandle,
        d_bias_a: &MatrixBufferHandle,
        d_wb: &MatrixBufferHandle,
        d_bias_b: &MatrixBufferHandle,
    ) -> Vec<f32> {
        assert!(
            x.is_gpu() && da.is_gpu() && db.is_gpu()
                && pre_a.is_gpu() && pre_b.is_gpu() && wa.is_gpu() && wb.is_gpu(),
            "Input handles must be GPU"
        );
        assert!(
            dx.is_gpu() && d_wa.is_gpu() && d_bias_a.is_gpu()
                && d_wb.is_gpu() && d_bias_b.is_gpu(),
            "Output handles must be GPU"
        );

        let batch = x.rows();
        let n = x.cols();
        let p = wa.rows();
        let q = wb.rows();

        assert_eq!(dx.rows(), batch);
        assert_eq!(dx.cols(), n);
        assert_eq!(d_wa.rows(), p);
        assert_eq!(d_wa.cols(), n);
        assert_eq!(d_bias_a.rows(), 1);
        assert_eq!(d_bias_a.cols(), p);
        assert_eq!(d_wb.rows(), q);
        assert_eq!(d_wb.cols(), n);
        assert_eq!(d_bias_b.rows(), 1);
        assert_eq!(d_bias_b.cols(), q);

        let x_buf = self.get_gpu_subbuffer_from_handle(x);
        let da_buf = self.get_gpu_subbuffer_from_handle(da);
        let db_buf = self.get_gpu_subbuffer_from_handle(db);
        let pre_a_buf = self.get_gpu_subbuffer_from_handle(pre_a);
        let pre_b_buf = self.get_gpu_subbuffer_from_handle(pre_b);
        let wa_buf = self.get_gpu_subbuffer_from_handle(wa);
        let wb_buf = self.get_gpu_subbuffer_from_handle(wb);
        let dx_buf = self.get_gpu_subbuffer_from_handle(dx);
        let d_wa_buf = self.get_gpu_subbuffer_from_handle(d_wa);
        let d_bias_a_buf = self.get_gpu_subbuffer_from_handle(d_bias_a);
        let d_wb_buf = self.get_gpu_subbuffer_from_handle(d_wb);
        let d_bias_b_buf = self.get_gpu_subbuffer_from_handle(d_bias_b);

        let pipelines = self.splitter_pipelines();
        let push = [batch as u32, n as u32, p as u32, q as u32];

        // 1. bwd_dx: градиент по входу.
        let total_dx = batch * n;
        self.run_compute_shader(
            &pipelines.backward_dx,
            &[
                (0, da_buf.clone()),
                (1, db_buf.clone()),
                (2, pre_a_buf.clone()),
                (3, pre_b_buf.clone()),
                (4, wa_buf.clone()),
                (5, wb_buf.clone()),
                (6, dx_buf.clone()),
            ],
            &push,
            total_dx,
        );

        // 2. bwd_params: градиенты весов и смещений.
        let total_params = p * n + q * n + p + q;
        self.run_compute_shader(
            &pipelines.backward_params,
            &[
                (0, x_buf),
                (1, da_buf),
                (2, db_buf),
                (3, pre_a_buf),
                (4, pre_b_buf),
                (5, d_wa_buf),
                (6, d_bias_a_buf),
                (7, d_wb_buf),
                (8, d_bias_b_buf),
            ],
            &push,
            total_params,
        );

        let mut grad = Vec::with_capacity(p * n + q * n + p + q);
        grad.extend_from_slice(&gpu_handle_to_vec(self, d_wa));
        grad.extend_from_slice(&gpu_handle_to_vec(self, d_wb));
        grad.extend_from_slice(&gpu_handle_to_vec(self, d_bias_a));
        grad.extend_from_slice(&gpu_handle_to_vec(self, d_bias_b));
        grad
    }
}

/// Вспомогательная функция: скачивает GPU-данные в CPU-буфер и возвращает Vec<f32>.
fn gpu_handle_to_vec(gpu: &GpuCompute, handle: &MatrixBufferHandle) -> Vec<f32> {
    let cpu_handle = gpu.download_gpu_handle_to_cpu_handle(handle);
    let guard = cpu_handle.read();
    guard.as_slice().unwrap().to_vec()
}