use crate::compute_manager::graph::types::DynamicContext;
use crate::linalg;
use faer::Mat;

pub struct CombinerConnector;

impl CombinerConnector {
    pub fn new(_input_dims: Vec<usize>) -> Self {
        Self
    }

    /// Прямой проход: возвращает входную матрицу без изменений.
    pub fn forward_mat(
        &self,
        input: &Mat<f32>,
    ) -> (Mat<f32>, DynamicContext) {
        let ctx = DynamicContext::Ctx1D(
            crate::layers::context1d::LayerContext1D::CombinerConnector {
                inputs: vec![linalg::faer_to_tensor2d(input)],
            },
        );
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

    pub fn param_len(&self) -> usize { 0 }
}