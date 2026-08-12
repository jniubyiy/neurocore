// src/compute_manager/gpu/compute/dim_ops.rs

use faer::Mat;
use super::base::GpuCompute;
use crate::compute_manager::dim_change;
use crate::compute_manager::matrix_buffer::MatrixBuffer;

impl GpuCompute {
    /// Выполняет операцию reduce (уменьшение размерности) над матрицей.
    /// Старая версия для обратной совместимости.
    pub fn run_reduce_mat(&self, mat: &Mat<f32>, target_dims: &[usize]) -> Mat<f32> {
        reduce_mat_gpu(self, mat, target_dims)
    }

    /// Выполняет операцию unsqueeze (увеличение размерности) над матрицей на GPU.
    /// Старая версия для обратной совместимости.
    pub fn run_unsqueeze_mat(&self, mat: &Mat<f32>, target_dims: &[usize]) -> Mat<f32> {
        unsqueeze_mat_gpu(self, mat, target_dims)
    }

    // -------------------------------------------------------------------------
    // НОВЫЕ БУФЕРИЗОВАННЫЕ ВЕРСИИ (MatrixBuffer)
    // -------------------------------------------------------------------------

    /// Выполняет операцию reduce над GPU MatrixBuffer.
    /// Вход и выход — GPU буферы. Реализовано через CPU fallback.
    pub fn run_reduce_mat_buffered(&self, input: &MatrixBuffer, target_dims: &[usize]) -> MatrixBuffer {
        assert!(input.is_gpu(), "Input buffer must be GPU");
        let mat = self.download_gpu_matrix_to_mat(input);
        let reduced = dim_change::reduce_mat(&mat, target_dims);
        self.upload_mat_to_gpu_matrix(&reduced)
    }

    /// Выполняет операцию unsqueeze над GPU MatrixBuffer с использованием шейдера.
    pub fn run_unsqueeze_mat_buffered(&self, input: &MatrixBuffer, target_dims: &[usize]) -> MatrixBuffer {
        assert!(input.is_gpu(), "Input buffer must be GPU");
        unsqueeze_mat_gpu_buffered(self, input, target_dims)
    }
}

/// GPU‑версия операции reduce (обратное преобразование по сравнению с unsqueeze).
/// На данный момент реализована через CPU fallback.
pub fn reduce_mat_gpu(gpu: &GpuCompute, mat: &Mat<f32>, target_dims: &[usize]) -> Mat<f32> {
    dim_change::reduce_mat(mat, target_dims)
}

/// GPU‑версия операции unsqueeze с использованием шейдера (старая версия с Mat).
pub fn unsqueeze_mat_gpu(gpu: &GpuCompute, mat: &Mat<f32>, target_dims: &[usize]) -> Mat<f32> {
    let total_elements = mat.nrows() * mat.ncols();
    let last_dim = target_dims[target_dims.len() - 1];
    let remaining_product: usize = target_dims[..target_dims.len() - 1].iter().product();

    assert_eq!(
        mat.ncols(),
        remaining_product * last_dim,
        "unsqueeze_mat_gpu: input columns must equal product of target_dims"
    );

    let batch = mat.nrows();
    let new_rows = batch * remaining_product;
    let new_cols = last_dim;

    let in_rows = mat.nrows() as u32;
    let in_cols = mat.ncols() as u32;
    let out_rows = new_rows as u32;
    let out_cols = new_cols as u32;

    let flat = GpuCompute::mat_to_flat(mat);
    let (in_buf, in_raw) = gpu.upload_to_temp_buffer(&flat);
    let (out_buf, out_raw) = gpu.acquire_temp_buffer(total_elements);

    let push = [in_rows, in_cols, out_rows, out_cols];
    gpu.run_compute_shader(
        gpu.pipeline_cache.unsqueeze.clone(),
        &[(0, in_buf.clone()), (1, out_buf.clone())],
        &push,
        total_elements,
    );

    let result = gpu.read_temp_buffer_to_mat(out_buf, out_raw, new_rows, new_cols);
    gpu.release_temp_buffer(in_buf, in_raw);
    result
}

/// Буферизованная версия unsqueeze, работающая напрямую с GPU MatrixBuffer.
pub fn unsqueeze_mat_gpu_buffered(gpu: &GpuCompute, input: &MatrixBuffer, target_dims: &[usize]) -> MatrixBuffer {
    assert!(input.is_gpu(), "Input buffer must be GPU");

    let total_elements = input.rows() * input.cols();
    let last_dim = target_dims[target_dims.len() - 1];
    let remaining_product: usize = target_dims[..target_dims.len() - 1].iter().product();

    assert_eq!(
        input.cols(),
        remaining_product * last_dim,
        "unsqueeze_mat_gpu_buffered: input columns must equal product of target_dims"
    );

    let batch = input.rows();
    let new_rows = batch * remaining_product;
    let new_cols = last_dim;

    let in_rows = input.rows() as u32;
    let in_cols = input.cols() as u32;
    let out_rows = new_rows as u32;
    let out_cols = new_cols as u32;

    let in_buf = input.as_gpu_buffer().expect("GPU buffer");
    let output = gpu.allocate_gpu_matrix(new_rows, new_cols);
    let out_buf = output.as_gpu_buffer().expect("GPU buffer");

    let push = [in_rows, in_cols, out_rows, out_cols];
    gpu.run_compute_shader(
        gpu.pipeline_cache.unsqueeze.clone(),
        &[(0, in_buf.clone()), (1, out_buf.clone())],
        &push,
        total_elements,
    );

    output
}