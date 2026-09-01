// src/layers/combiner/gpu/mod.rs

pub mod pipeline;   // <-- новый модуль

use crate::compute_manager::gpu::compute::GpuCompute;
use crate::compute_manager::matrix_buffer::MatrixBufferHandle;

impl GpuCompute {
    pub fn run_combiner_forward_buffered_handle(
        &self,
        a: &MatrixBufferHandle,
        b: &MatrixBufferHandle,
        wa: &MatrixBufferHandle,
        wb: &MatrixBufferHandle,
        bias: &[f32],
        out: &MatrixBufferHandle,
        pre: &MatrixBufferHandle,
    ) {
        assert!(a.is_gpu() && b.is_gpu() && wa.is_gpu() && wb.is_gpu(), "Input handles must be GPU");
        assert!(out.is_gpu() && pre.is_gpu(), "Output handles must be GPU");

        let batch = a.rows();
        let n = a.cols();
        let m = wa.rows();
        assert_eq!(b.rows(), batch);
        assert_eq!(b.cols(), n);
        assert_eq!(out.rows(), batch);
        assert_eq!(out.cols(), m);
        assert_eq!(pre.rows(), batch);
        assert_eq!(pre.cols(), m);

        let a_buf = self.get_gpu_subbuffer_from_handle(a);
        let b_buf = self.get_gpu_subbuffer_from_handle(b);
        let wa_buf = self.get_gpu_subbuffer_from_handle(wa);
        let wb_buf = self.get_gpu_subbuffer_from_handle(wb);
        let out_buf = self.get_gpu_subbuffer_from_handle(out);
        let pre_buf = self.get_gpu_subbuffer_from_handle(pre);

        let (bias_buf, bias_raw) = self.upload_to_temp_buffer(bias);

        // Используем новый пайплайн из собственной структуры Combiner
        let pipeline = &self.combiner_pipelines().forward;
        let push = [batch as u32, n as u32, m as u32];
        self.run_compute_shader_with_dispatch(
            pipeline,
            &[
                (0, a_buf),
                (1, b_buf),
                (2, wa_buf),
                (3, wb_buf),
                (4, bias_buf.clone()),
                (5, out_buf),
                (6, pre_buf),
            ],
            &push,
            [((batch + 255) / 256) as u32, 1, 1],
        );

        self.release_temp_buffer(bias_buf, bias_raw);
    }

    pub fn run_combiner_backward_buffered_handle(
        &self,
        a: &MatrixBufferHandle,
        b: &MatrixBufferHandle,
        d_out: &MatrixBufferHandle,
        pre: &MatrixBufferHandle,
        wa: &MatrixBufferHandle,
        wb: &MatrixBufferHandle,
        da: &MatrixBufferHandle,
        db: &MatrixBufferHandle,
        d_wa: &MatrixBufferHandle,
        d_wb: &MatrixBufferHandle,
        d_bias: &MatrixBufferHandle,
    ) -> Vec<f32> {
        assert!(a.is_gpu() && b.is_gpu() && d_out.is_gpu() && pre.is_gpu() && wa.is_gpu() && wb.is_gpu(),
            "Input handles must be GPU");
        assert!(da.is_gpu() && db.is_gpu() && d_wa.is_gpu() && d_wb.is_gpu() && d_bias.is_gpu(),
            "Output handles must be GPU");

        let batch = a.rows();
        let n = a.cols();
        let m = wa.rows();
        assert_eq!(b.rows(), batch);
        assert_eq!(b.cols(), n);
        assert_eq!(d_out.rows(), batch);
        assert_eq!(d_out.cols(), m);
        assert_eq!(pre.rows(), batch);
        assert_eq!(pre.cols(), m);
        assert_eq!(da.rows(), batch);
        assert_eq!(da.cols(), n);
        assert_eq!(db.rows(), batch);
        assert_eq!(db.cols(), n);
        assert_eq!(d_wa.rows(), m);
        assert_eq!(d_wa.cols(), n);
        assert_eq!(d_wb.rows(), m);
        assert_eq!(d_wb.cols(), n);
        assert_eq!(d_bias.rows(), 1);
        assert_eq!(d_bias.cols(), m);

        let a_buf = self.get_gpu_subbuffer_from_handle(a);
        let b_buf = self.get_gpu_subbuffer_from_handle(b);
        let d_out_buf = self.get_gpu_subbuffer_from_handle(d_out);
        let pre_buf = self.get_gpu_subbuffer_from_handle(pre);
        let wa_buf = self.get_gpu_subbuffer_from_handle(wa);
        let wb_buf = self.get_gpu_subbuffer_from_handle(wb);
        let da_buf = self.get_gpu_subbuffer_from_handle(da);
        let db_buf = self.get_gpu_subbuffer_from_handle(db);
        let d_wa_buf = self.get_gpu_subbuffer_from_handle(d_wa);
        let d_wb_buf = self.get_gpu_subbuffer_from_handle(d_wb);
        let d_bias_buf = self.get_gpu_subbuffer_from_handle(d_bias);

        self.fill_gpu_handle(d_wa, 0.0);
        self.fill_gpu_handle(d_wb, 0.0);
        self.fill_gpu_handle(d_bias, 0.0);

        // Используем новый пайплайн из собственной структуры Combiner
        let pipeline = &self.combiner_pipelines().backward;
        let push = [batch as u32, n as u32, m as u32];
        self.run_compute_shader_with_dispatch(
            pipeline,
            &[
                (0, d_out_buf),
                (1, pre_buf),
                (2, a_buf),
                (3, b_buf),
                (4, wa_buf),
                (5, wb_buf),
                (6, da_buf),
                (7, db_buf),
                (8, d_wa_buf),
                (9, d_wb_buf),
                (10, d_bias_buf),
            ],
            &push,
            [((batch + 255) / 256) as u32, 1, 1],
        );

        let mut grad = Vec::with_capacity(2 * m * n + m);
        grad.extend_from_slice(&gpu_handle_to_vec(self, d_wa));
        grad.extend_from_slice(&gpu_handle_to_vec(self, d_wb));
        grad.extend_from_slice(&gpu_handle_to_vec(self, d_bias));
        grad
    }
}

/// Вспомогательная функция: скачивает GPU-данные в CPU-буфер и возвращает Vec<f32>.
fn gpu_handle_to_vec(gpu: &GpuCompute, handle: &MatrixBufferHandle) -> Vec<f32> {
    let cpu_handle = gpu.download_gpu_handle_to_cpu_handle(handle);
    let guard = cpu_handle.read();
    guard.as_slice().unwrap().to_vec()
}