// src/layers/splitter_connector/splitter_connector.rs

use crate::compute_manager::graph::types::DynamicContext;
use crate::layers::mat_context::MatContext;
use faer::Mat;

pub struct SplitterConnector {
    pub dim_a: usize,
    pub dim_b: usize,
}

impl SplitterConnector {
    pub fn new(dim_a: usize, dim_b: usize) -> Self {
        Self { dim_a, dim_b }
    }

    /// Прямой проход: принимает две матрицы и возвращает их же,
    /// сохраняя входную матрицу `a` в матричном контексте для обратного прохода.
    pub fn forward_mat(
        &self,
        input_a: &Mat<f32>,
        input_b: &Mat<f32>,
    ) -> (Mat<f32>, Mat<f32>, DynamicContext) {
        let ctx = DynamicContext::Mat(MatContext::SplitterConnector {
            input: input_a.clone(),
        });
        (input_a.clone(), input_b.clone(), ctx)
    }

    /// Обратный проход: градиенты проходят насквозь без изменений.
    /// Контекст не используется, оставлен для совместимости с сигнатурой вызова.
    pub fn backward_mat(
        &self,
        _ctx: &DynamicContext,
        delta_a: &Mat<f32>,
        delta_b: &Mat<f32>,
    ) -> (Mat<f32>, Mat<f32>, Vec<f32>) {
        (delta_a.clone(), delta_b.clone(), vec![])
    }

    pub fn param_len(&self) -> usize {
        0
    }
}