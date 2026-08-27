// src/compute_manager/gpu/compute/matmul.rs

use super::base::GpuCompute;
use crate::compute_manager::matrix_buffer::MatrixBufferHandle;

impl GpuCompute {
    /// Умножение матриц A (M×K) и B (K×N) → C (M×N) на GPU.
    /// Handle-версия: все входы/выходы — GPU-дескрипторы.
    /// Результат записывается в предоставленный `out`.
    pub fn run_mat_mul_buffered_handle(
        &self,
        a: &MatrixBufferHandle,
        b: &MatrixBufferHandle,
        out: &MatrixBufferHandle,
    ) {
        let m = a.rows();
        let n = b.cols();
        let k = a.cols();
        assert_eq!(k, b.rows(), "MatMul dimensions mismatch");
        assert!(a.is_gpu(), "a must be GPU");
        assert!(b.is_gpu(), "b must be GPU");
        assert!(out.is_gpu(), "out must be GPU");
        assert_eq!(out.rows(), m, "out rows mismatch");
        assert_eq!(out.cols(), n, "out cols mismatch");

        let a_buf = self.get_gpu_subbuffer_from_handle(a);
        let b_buf = self.get_gpu_subbuffer_from_handle(b);
        let out_buf = self.get_gpu_subbuffer_from_handle(out);

        let pipeline = self.pipeline_cache.mat_mul_pipeline();
        let push: [u32; 3] = [m as u32, n as u32, k as u32];
        let dispatch_dim = [
            ((m + 15) / 16) as u32,
            ((n + 15) / 16) as u32,
            1u32,
        ];

        self.run_compute_shader_with_dispatch(
            pipeline,
            &[(0, a_buf), (1, b_buf), (2, out_buf)],
            &push,
            dispatch_dim,
        );
    }

    /// Суммирование строк матрицы → вектор длиной `cols`.
    /// Handle-версия: входной GPU-дескриптор, выходной GPU-дескриптор размера `(1, cols)`.
    /// Возвращает `Vec<f32>` для удобства (например, для градиентов bias).
    pub fn run_reduce_sum_cols_handle(
        &self,
        input: &MatrixBufferHandle,
        output: &MatrixBufferHandle,
    ) -> Vec<f32> {
        assert!(input.is_gpu() && output.is_gpu(), "Handles must be GPU");
        let rows = input.rows();
        let cols = input.cols();
        assert_eq!(output.rows(), 1, "output must have one row");
        assert_eq!(output.cols(), cols, "output cols mismatch");

        let in_buf = self.get_gpu_subbuffer_from_handle(input);
        let out_buf = self.get_gpu_subbuffer_from_handle(output);

        let pipeline = self.pipeline_cache.reduce_pipeline();
        let push: [u32; 1] = [rows as u32];

        self.run_compute_shader_with_dispatch(
            pipeline,
            &[(0, in_buf), (1, out_buf)],
            &push,
            [cols as u32, 1, 1], // одна workgroup на столбец
        );

        // Читаем результат обратно в CPU через управляемый буфер.
        let cpu_handle = self.download_gpu_handle_to_cpu_handle(output);
        let guard = cpu_handle.read();
        guard.as_slice().unwrap().to_vec()
    }
}