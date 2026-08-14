// src/compute_manager/gpu/compute/dim_ops.rs

use faer::Mat;
use super::base::GpuCompute;
use crate::compute_manager::dim_change;
use crate::compute_manager::matrix_buffer::MatrixBuffer;
use crate::compute_manager::matrix_buffer::MatrixBufferHandle;

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
    // Буферизованные версии (MatrixBuffer)
    // -------------------------------------------------------------------------

    /// Выполняет операцию reduce над GPU MatrixBuffer.
    /// Вход и выход — GPU буферы. Реализовано через CPU-стагинг без `faer::Mat`.
    pub fn run_reduce_mat_buffered(&self, input: &MatrixBuffer, target_dims: &[usize]) -> MatrixBuffer {
        assert!(input.is_gpu(), "Input buffer must be GPU");

        let total = input.rows() * input.cols();
        let remaining_product: usize = target_dims[..target_dims.len()-1].iter().product();
        let batch = input.rows() / remaining_product;
        let new_rows = batch;
        let new_cols = total / new_rows;

        assert_eq!(total, new_rows * new_cols, "reduce_mat_buffered: element count mismatch");

        // Скачиваем данные в Vec (column-major исходной матрицы)
        let old_vec = self.download_gpu_matrix_to_vec(input);
        let mut new_vec = vec![0.0f32; total];

        // Переупаковка из column-major старой формы в column-major новой формы
        for idx in 0..total {
            let dst_r = idx / new_cols;
            let dst_c = idx % new_cols;
            let new_idx = dst_c * new_rows + dst_r;
            new_vec[new_idx] = old_vec[idx];
        }

        // Загружаем результат обратно на GPU
        self.upload_vec_to_gpu_buffer(&new_vec, new_rows, new_cols)
    }

    /// Выполняет операцию unsqueeze над GPU MatrixBuffer с использованием шейдера.
    pub fn run_unsqueeze_mat_buffered(&self, input: &MatrixBuffer, target_dims: &[usize]) -> MatrixBuffer {
        assert!(input.is_gpu(), "Input buffer must be GPU");
        unsqueeze_mat_gpu_buffered(self, input, target_dims)
    }

    // -------------------------------------------------------------------------
    // Handle-версии (MatrixBufferHandle)
    // -------------------------------------------------------------------------

    /// Выполняет операцию reduce (уменьшение размерности) над GPU-дескриптором.
    /// Вход и выход — GPU-дескрипторы. Реализовано через CPU-стагинг без `faer::Mat`.
    pub fn run_reduce_mat_buffered_handle(
        &self,
        input: &MatrixBufferHandle,
        target_dims: &[usize],
        output: &MatrixBufferHandle,
    ) {
        assert!(input.is_gpu(), "Input handle must be GPU");
        assert!(output.is_gpu(), "Output handle must be GPU");

        let total = input.rows() * input.cols();
        let remaining_product: usize = target_dims[..target_dims.len()-1].iter().product();
        let batch = input.rows() / remaining_product;
        let new_rows = batch;
        let new_cols = total / new_rows;

        assert_eq!(total, new_rows * new_cols, "reduce_mat_buffered_handle: element count mismatch");
        assert_eq!(output.rows(), new_rows, "Output rows mismatch");
        assert_eq!(output.cols(), new_cols, "Output cols mismatch");

        // Скачиваем данные в Vec (column-major исходной матрицы)
        let old_vec = self.download_gpu_handle_to_vec(input);

        // Переупаковка из column-major старой формы в column-major новой формы
        let mut new_vec = vec![0.0f32; total];
        for idx in 0..total {
            let dst_r = idx / new_cols;
            let dst_c = idx % new_cols;
            let new_idx = dst_c * new_rows + dst_r;
            new_vec[new_idx] = old_vec[idx];
        }

        // Загружаем результат в выходной GPU-дескриптор
        self.copy_slice_to_gpu_handle(output, &new_vec);
    }

    /// Выполняет операцию unsqueeze (увеличение размерности) над GPU-дескриптором.
    /// Вход и выход — GPU-дескрипторы. Используется шейдер unsqueeze.
    pub fn run_unsqueeze_mat_buffered_handle(
        &self,
        input: &MatrixBufferHandle,
        target_dims: &[usize],
        output: &MatrixBufferHandle,
    ) {
        assert!(input.is_gpu(), "Input handle must be GPU");
        assert!(output.is_gpu(), "Output handle must be GPU");

        let total_elements = input.rows() * input.cols();
        let last_dim = target_dims[target_dims.len() - 1];
        let remaining_product: usize = target_dims[..target_dims.len() - 1].iter().product();

        assert_eq!(
            input.cols(),
            remaining_product * last_dim,
            "unsqueeze_mat_buffered_handle: input columns must equal product of target_dims"
        );

        let batch = input.rows();
        let new_rows = batch * remaining_product;
        let new_cols = last_dim;

        assert_eq!(output.rows(), new_rows, "Output rows mismatch");
        assert_eq!(output.cols(), new_cols, "Output cols mismatch");

        let in_rows = input.rows() as u32;
        let in_cols = input.cols() as u32;
        let out_rows = new_rows as u32;
        let out_cols = new_cols as u32;

        let in_buf = self.get_gpu_subbuffer_from_handle(input);
        let out_buf = self.get_gpu_subbuffer_from_handle(output);

        let push = [in_rows, in_cols, out_rows, out_cols];
        self.run_compute_shader(
            self.pipeline_cache.unsqueeze.clone(),
            &[(0, in_buf), (1, out_buf)],
            &push,
            total_elements,
        );
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