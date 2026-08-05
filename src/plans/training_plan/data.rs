// src/training_plan/data.rs

use crate::compute_manager::dim_change::DynamicTensor;
use crate::tensor::{Tensor2D, Tensor3D, Tensor4D, Tensor5D};
use super::plan::DataSource;

impl DataSource {
    /// Количество примеров (размер по первому измерению).
    pub fn num_samples(&self) -> usize {
        match self {
            DataSource::Tensor2D(t) => t.dim1,
            DataSource::Tensor3D(t) => t.dim1,
            DataSource::Tensor4D(t) => t.dim1,
            DataSource::Tensor5D(t) => t.dim1,
        }
    }

    /// Возвращает полные размерности тензора, включая batch.
    pub fn dimensions(&self) -> Vec<usize> {
        match self {
            DataSource::Tensor2D(t) => vec![t.dim1, t.dim2],
            DataSource::Tensor3D(t) => vec![t.dim1, t.dim2, t.dim3],
            DataSource::Tensor4D(t) => vec![t.dim1, t.dim2, t.dim3, t.dim4],
            DataSource::Tensor5D(t) => vec![t.dim1, t.dim2, t.dim3, t.dim4, t.dim5],
        }
    }

    /// Преобразует весь датасет в DynamicTensor нужной размерности.
    pub fn to_dynamic_tensor(&self) -> DynamicTensor {
        match self {
            DataSource::Tensor2D(t) => DynamicTensor::Dim1(t.clone()),
            DataSource::Tensor3D(t) => DynamicTensor::Dim2(t.clone()),
            DataSource::Tensor4D(t) => DynamicTensor::Dim3(t.clone()),
            DataSource::Tensor5D(t) => DynamicTensor::Dim4(t.clone()),
        }
    }

    /// Извлекает подтензор по индексам `[start, end)` вдоль первого измерения (batch).
    pub fn batch(&self, start: usize, end: usize) -> DynamicTensor {
        match self {
            DataSource::Tensor2D(t) => {
                let rows: Vec<Vec<f32>> = t.data[start..end].to_vec();
                DynamicTensor::Dim1(Tensor2D::new(rows))
            }
            DataSource::Tensor3D(t) => {
                let slice = t.data[start..end].to_vec();
                DynamicTensor::Dim2(Tensor3D::new(slice))
            }
            DataSource::Tensor4D(t) => {
                let slice = t.data[start..end].to_vec();
                DynamicTensor::Dim3(Tensor4D::new(slice))
            }
            DataSource::Tensor5D(t) => {
                let slice = t.data[start..end].to_vec();
                DynamicTensor::Dim4(Tensor5D::new(slice))
            }
        }
    }
}