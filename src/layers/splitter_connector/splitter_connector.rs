// src/layers/splitter_connector/splitter_connector.rs

use crate::compute_manager::graph::types::DynamicContext;
use crate::linalg;
use faer::Mat;

pub struct SplitterConnector {
    pub dim_a: usize,
    pub dim_b: usize,
}

impl SplitterConnector {
    pub fn new(dim_a: usize, dim_b: usize) -> Self {
        Self { dim_a, dim_b }
    }

    /// Прямой проход: принимает две матрицы и возвращает их же.
    pub fn forward_mat(
        &self,
        input_a: &Mat<f32>,
        input_b: &Mat<f32>,
    ) -> (Mat<f32>, Mat<f32>, DynamicContext) {
        let ctx = DynamicContext::Ctx1D(
            crate::layers::context1d::LayerContext1D::SplitterConnector {
                input: linalg::faer_to_tensor2d(input_a),
            },
        );
        (input_a.clone(), input_b.clone(), ctx)
    }

    /// Обратный проход: просто пробрасывает градиенты обратно.
    pub fn backward_mat(
        &self,
        _ctx: &DynamicContext,
        delta_a: &Mat<f32>,
        delta_b: &Mat<f32>,
    ) -> (Mat<f32>, Mat<f32>, Vec<f32>) {
        (delta_a.clone(), delta_b.clone(), vec![])
    }

    pub fn param_len(&self) -> usize { 0 }
}