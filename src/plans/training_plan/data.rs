// src/training_plan/data.rs

use crate::tensor::Tensor2D;
use super::plan::DataSource;

impl DataSource {
    /// Создаёт DataSource из Tensor2D.
    pub fn from_tensor2d(tensor: Tensor2D) -> Self {
        DataSource::Tensor2D(tensor)
    }

    /// Количество примеров.
    pub fn num_samples(&self) -> usize {
        match self {
            DataSource::Tensor2D(t) => t.dim1,
        }
    }

    /// Разбивает данные на батчи заданного размера.
    /// Возвращает вектор батчей (каждый батч — Tensor2D).
    pub fn batches(&self, batch_size: usize) -> Vec<Tensor2D> {
        match self {
            DataSource::Tensor2D(tensor) => {
                let n = tensor.dim1;
                let mut batches = Vec::new();
                for start in (0..n).step_by(batch_size) {
                    let end = (start + batch_size).min(n);
                    let rows: Vec<Vec<f32>> = tensor.data[start..end].to_vec();
                    batches.push(Tensor2D::new(rows));
                }
                batches
            }
        }
    }
}