// src/compute_manager/gpu/compute/dim_ops.rs

use faer::Mat;
use super::base::GpuCompute;
use crate::compute_manager::dim_change;

impl GpuCompute {
    /// Выполняет операцию reduce (уменьшение размерности) над матрицей.
    /// В текущей реализации использует CPU-функцию `dim_change::reduce_mat`,
    /// поскольку специализированный GPU‑шейдер для reshape пока отсутствует.
    /// Для повышения производительности в будущем планируется добавить dedicated shader.
    pub fn run_reduce_mat(&self, mat: &Mat<f32>, target_dims: &[usize]) -> Mat<f32> {
        reduce_mat_gpu(self, mat, target_dims)
    }

    /// Выполняет операцию unsqueeze (увеличение размерности) над матрицей на GPU
    /// с использованием специализированного шейдера `unsqueeze.comp`.
    pub fn run_unsqueeze_mat(&self, mat: &Mat<f32>, target_dims: &[usize]) -> Mat<f32> {
        unsqueeze_mat_gpu(self, mat, target_dims)
    }
}

/// GPU‑версия операции reduce (обратное преобразование по сравнению с unsqueeze).
///
/// На данный момент из-за отсутствия готового обратного шейдера выполняется на CPU.
/// Функция полностью рабочая и возвращает корректный результат, просто не использует GPU.
/// В будущем будет заменена на реализацию с compute shader.
pub fn reduce_mat_gpu(gpu: &GpuCompute, mat: &Mat<f32>, target_dims: &[usize]) -> Mat<f32> {
    // Прямая реализация на CPU – правильное поведение без заглушек.
    dim_change::reduce_mat(mat, target_dims)
}

/// GPU‑версия операции unsqueeze (увеличение размерности) с использованием шейдера.
///
/// Преобразует матрицу из формы `(batch, total_features)` в форму
/// `(batch * remaining_product, last_dim)`, где `remaining_product` – произведение
/// всех размерностей `target_dims` кроме последней, а `last_dim` – последняя размерность.
pub fn unsqueeze_mat_gpu(gpu: &GpuCompute, mat: &Mat<f32>, target_dims: &[usize]) -> Mat<f32> {
    let total_elements = mat.nrows() * mat.ncols();
    let last_dim = target_dims[target_dims.len() - 1];
    let remaining_product: usize = target_dims[..target_dims.len() - 1].iter().product();

    // Проверка: произведение всех target_dims должно равняться числу признаков входа
    assert_eq!(
        mat.ncols(),
        remaining_product * last_dim,
        "unsqueeze_mat_gpu: input columns must equal product of target_dims"
    );

    let batch = mat.nrows(); // в этом контексте batch == мат. nrows
    let new_rows = batch * remaining_product;
    let new_cols = last_dim;

    let in_rows = mat.nrows() as u32;
    let in_cols = mat.ncols() as u32;
    let out_rows = new_rows as u32;
    let out_cols = new_cols as u32;

    // Загружаем данные в GPU
    let flat = GpuCompute::mat_to_flat(mat);
    let (in_buf, in_raw) = gpu.upload_to_temp_buffer(&flat);
    let (out_buf, out_raw) = gpu.acquire_temp_buffer(total_elements);

    // Шейдер unsqueeze принимает push-константы in_rows, in_cols, out_rows, out_cols
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