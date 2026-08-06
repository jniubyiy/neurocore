// src/layers/combiner_connector/combiner_connector.rs

use crate::compute_manager::graph::types::DynamicContext;
use crate::layers::mat_context::MatContext;
use faer::Mat;

pub struct CombinerConnector;

impl CombinerConnector {
    pub fn new(_input_dims: Vec<usize>) -> Self {
        Self
    }

    /// Прямой проход: возвращает входную матрицу без изменений,
    /// сохраняя её в матричном контексте для потенциального использования в обратном проходе.
    pub fn forward_mat(
        &self,
        input: &Mat<f32>,
    ) -> (Mat<f32>, DynamicContext) {
        let ctx = DynamicContext::Mat(MatContext::CombinerConnector {
            inputs: vec![input.clone()],
        });
        (input.clone(), ctx)
    }

    /// Обратный проход: градиент проходит насквозь.
    pub fn backward_mat(
        &self,
        _ctx: &DynamicContext,
        delta: &Mat<f32>,
    ) -> (Mat<f32>, Vec<f32>) {
        (delta.clone(), vec![])
    }

    pub fn param_len(&self) -> usize {
        0
    }
}