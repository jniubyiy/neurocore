// src/compute_manager/gpu/compute/matmul.rs

use faer::Mat;
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::{Pipeline, PipelineBindPoint};
use super::base::GpuCompute;

impl GpuCompute {
    /// Умножение матриц A (M×K) и B (K×N) → C (M×N) на GPU с тайловым шейдером.
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

    /// Суммирование строк матрицы → вектор длиной `cols`.
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
}