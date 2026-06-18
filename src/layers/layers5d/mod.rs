// src/layers/layers5d/mod.rs

use crate::tensor::Tensor5D;
use crate::model_plan::param_store::ParamSlice;

pub mod linear5d;
pub mod relu5d;
pub mod sigmoid5d;
pub mod softmax5d;
pub mod tanh5d;
pub mod memory5d;

pub use linear5d::Linear5D;
pub use relu5d::ReLU5D;
pub use sigmoid5d::Sigmoid5D;
pub use softmax5d::Softmax5D;
pub use tanh5d::Tanh5D;
pub use memory5d::Memory5D;

#[derive(Clone)]
pub enum LayerContext5D {
    Linear5D  { input: Tensor5D },
    ReLU5D    { input: Tensor5D },
    Sigmoid5D { output: Tensor5D },
    Tanh5D    { output: Tensor5D },
    Softmax5D { output: Tensor5D },
    Memory5D  { input: Tensor5D },
}

pub trait Layer5D: Send + Sync {
    /// Размер пятой оси (последней) входного тензора (для совместимости).
    fn dim5(&self) -> usize { self.input_dims()[0] }

    /// Размер пятой оси (последней) выходного тензора (для совместимости).
    fn out_dim5(&self) -> usize { self.output_dims()[0] }

    /// Размерности каждого из входных тензоров (последняя ось).
    fn input_dims(&self) -> Vec<usize>;

    /// Размерности каждого из выходных тензоров (последняя ось).
    fn output_dims(&self) -> Vec<usize>;

    /// Прямой проход с созданием нового тензора.
    fn forward(&self, inputs: &[Tensor5D], params: &[f32], slice: &ParamSlice) -> (Vec<Tensor5D>, Vec<LayerContext5D>) {
        let out_sizes = self.output_dims();
        let dim1 = inputs.first().map(|t| t.dim1).unwrap_or(0);
        let dim2 = inputs.first().map(|t| t.dim2).unwrap_or(0);
        let dim3 = inputs.first().map(|t| t.dim3).unwrap_or(0);
        let dim4 = inputs.first().map(|t| t.dim4).unwrap_or(0);
        let mut out_bufs: Vec<Vec<Vec<Vec<Vec<Vec<f32>>>>>> = out_sizes
            .iter()
            .map(|&sz| vec![vec![vec![vec![vec![0.0; sz]; dim4]; dim3]; dim2]; dim1])
            .collect();
        let ctxs = self.forward_into(inputs, params, slice, &mut out_bufs);
        let tensors = out_bufs.into_iter().map(Tensor5D::new).collect();
        (tensors, ctxs)
    }

    /// Прямой проход с записью результата в предоставленные пятимерные буферы.
    fn forward_into(
        &self,
        inputs: &[Tensor5D],
        params: &[f32],
        slice: &ParamSlice,
        out_bufs: &mut [Vec<Vec<Vec<Vec<Vec<f32>>>>>],
    ) -> Vec<LayerContext5D>;

    /// Обратный проход: принимает градиенты по выходам (по одному на каждый выход),
    /// контексты (по одному на каждый выход) и возвращает:
    ///   - градиенты по входам (по одному на каждый вход)
    ///   - градиенты параметров слоя (плоский вектор)
    fn backward(
        &self,
        ctxs: &[LayerContext5D],
        deltas: &[Tensor5D],
        params: &[f32],
        slice: &ParamSlice,
    ) -> (Vec<Tensor5D>, Vec<f32>);

    /// Количество обучаемых параметров слоя.
    fn param_len(&self) -> usize;
}




