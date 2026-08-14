// src/compute_manager/gpu/compute/matmul.rs

use faer::Mat;
use super::base::GpuCompute;
use crate::compute_manager::matrix_buffer::MatrixBuffer;
use crate::compute_manager::matrix_buffer::MatrixBufferHandle;

impl GpuCompute {
    /// Умножение матриц A (M×K) и B (K×N) → C (M×N) на GPU с тайловым шейдером.
    /// Старая версия для обратной совместимости: принимает `Mat`, возвращает `Mat`.
    pub fn run_mat_mul(&self, a: &Mat<f32>, b: &Mat<f32>) -> Mat<f32> {
        let m = a.nrows();
        let n = b.ncols();
        let k = a.ncols();
        assert_eq!(k, b.nrows(), "MatMul dimensions mismatch");

        // Загружаем матрицы в GPU
        let a_data = Self::mat_to_flat(a);
        let b_data = Self::mat_to_flat(b);

        let (a_buf, a_raw) = self.upload_to_temp_buffer(&a_data);
        let (b_buf, b_raw) = self.upload_to_temp_buffer(&b_data);

        // Выходной буфер
        let total_out = m * n;
        let (out_buf, out_raw) = self.acquire_temp_buffer(total_out);

        let pipeline = self.pipeline_cache.mat_mul_pipeline();

        // Двумерный диспатч как в оригинале
        let dispatch_dim = [
            ((m + 15) / 16) as u32,
            ((n + 15) / 16) as u32,
            1u32,
        ];
        let push: [u32; 3] = [m as u32, n as u32, k as u32];

        self.run_compute_shader_2d(
            pipeline,
            &[(0, a_buf.clone()), (1, b_buf.clone()), (2, out_buf.clone())],
            &push,
            dispatch_dim,
        );

        // Читаем результат и освобождаем выходной буфер
        let result = self.read_temp_buffer_to_mat(out_buf, out_raw, m, n);

        // Возвращаем входные буферы в пул
        self.release_temp_buffer(a_buf, a_raw);
        self.release_temp_buffer(b_buf, b_raw);

        result
    }

    /// Умножение матриц A (M×K) и B (K×N) → C (M×N) на GPU.
    /// Новая буферизованная версия: принимает `MatrixBuffer` (GPU) и возвращает `MatrixBuffer` (GPU).
    pub fn run_mat_mul_buffered(&self, a: &MatrixBuffer, b: &MatrixBuffer) -> MatrixBuffer {
        let m = a.rows();
        let n = b.cols();
        let k = a.cols();
        assert_eq!(k, b.rows(), "MatMul dimensions mismatch");
        assert!(a.is_gpu() && b.is_gpu(), "Input buffers must be GPU");

        let a_buf = a.as_gpu_buffer().expect("a must be GPU");
        let b_buf = b.as_gpu_buffer().expect("b must be GPU");

        let out = self.allocate_gpu_matrix(m, n);
        let out_buf = out.as_gpu_buffer().expect("out must be GPU");

        let pipeline = self.pipeline_cache.mat_mul_pipeline();
        let push: [u32; 3] = [m as u32, n as u32, k as u32];
        let dispatch_dim = [
            ((m + 15) / 16) as u32,
            ((n + 15) / 16) as u32,
            1u32,
        ];

        self.run_compute_shader_with_dispatch(
            pipeline,
            &[(0, a_buf.clone()), (1, b_buf.clone()), (2, out_buf.clone())],
            &push,
            dispatch_dim,
        );

        out
    }

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
    /// Старая версия для обратной совместимости.
    pub fn run_reduce_sum_cols(&self, mat: &Mat<f32>) -> Vec<f32> {
        let rows = mat.nrows();
        let cols = mat.ncols();

        // Загружаем матрицу в GPU
        let data = Self::mat_to_flat(mat);
        let (in_buf, in_raw) = self.upload_to_temp_buffer(&data);

        // Выходной буфер для сумм по столбцам
        let (out_buf, out_raw) = self.acquire_temp_buffer(cols);

        let pipeline = self.pipeline_cache.reduce_pipeline();

        // Этот шейдер предполагает одномерный диспатч (по числу столбцов), поэтому используем run_compute_shader
        let push: [u32; 1] = [rows as u32];
        self.run_compute_shader(
            pipeline,
            &[(0, in_buf.clone()), (1, out_buf.clone())],
            &push,
            cols, // total_elements = cols
        );

        // Читаем результат через staging и возвращаем буферы
        let (staging_buf, staging_raw) = self.acquire_staging_buffer(cols);
        self.copy_buffer_sync(out_buf.clone(), staging_buf.clone());

        let result_vec = {
            let guard = staging_buf.read().expect("read staging");
            guard[..cols].to_vec()
        };

        // Возвращаем staging и выходной буфер
        self.release_staging_buffer(staging_buf, staging_raw);
        self.release_temp_buffer(out_buf, out_raw);

        // Возвращаем входной буфер
        self.release_temp_buffer(in_buf, in_raw);

        result_vec
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

        // Читаем результат обратно в CPU для возврата
        self.download_gpu_handle_to_vec(output)
    }
}