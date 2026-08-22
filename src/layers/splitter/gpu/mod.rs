// src/layers/splitter/gpu/mod.rs

use crate::compute_manager::gpu::compute::GpuCompute;
use crate::compute_manager::matrix_buffer::MatrixBufferHandle;

impl GpuCompute {
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
        assert!(out_a.is_gpu() && pre_a.is_gpu() && out_b.is_gpu() && pre_b.is_gpu(), "Output handles must be GPU");

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

        let pipeline = self.pipeline_cache.splitter_fwd.clone();
        let push = [batch as u32, n as u32, p as u32, q as u32];
        self.run_compute_shader_with_dispatch(
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
            [((batch + 255) / 256) as u32, 1, 1],
        );

        self.release_temp_buffer(bias_a_buf, bias_a_raw);
        self.release_temp_buffer(bias_b_buf, bias_b_raw);
    }

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
        assert!(x.is_gpu() && da.is_gpu() && db.is_gpu() && pre_a.is_gpu() && pre_b.is_gpu() && wa.is_gpu() && wb.is_gpu(),
            "Input handles must be GPU");
        assert!(dx.is_gpu() && d_wa.is_gpu() && d_bias_a.is_gpu() && d_wb.is_gpu() && d_bias_b.is_gpu(),
            "Output handles must be GPU");

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

        self.fill_gpu_handle(d_wa, 0.0);
        self.fill_gpu_handle(d_wb, 0.0);
        self.fill_gpu_handle(d_bias_a, 0.0);
        self.fill_gpu_handle(d_bias_b, 0.0);

        let pipeline = self.pipeline_cache.splitter_bwd.clone();
        let push = [batch as u32, n as u32, p as u32, q as u32];
        self.run_compute_shader_with_dispatch(
            pipeline,
            &[
                (0, x_buf),
                (1, da_buf),
                (2, db_buf),
                (3, pre_a_buf),
                (4, pre_b_buf),
                (5, wa_buf),
                (6, wb_buf),
                (7, dx_buf),
                (8, d_wa_buf),
                (9, d_bias_a_buf),
                (10, d_wb_buf),
                (11, d_bias_b_buf),
            ],
            &push,
            [((batch + 255) / 256) as u32, 1, 1],
        );

        let mut grad = Vec::with_capacity(p * n + q * n + p + q);
        grad.extend_from_slice(&self.download_gpu_handle_to_vec(d_wa));
        grad.extend_from_slice(&self.download_gpu_handle_to_vec(d_wb));
        grad.extend_from_slice(&self.download_gpu_handle_to_vec(d_bias_a));
        grad.extend_from_slice(&self.download_gpu_handle_to_vec(d_bias_b));
        grad
    }
}