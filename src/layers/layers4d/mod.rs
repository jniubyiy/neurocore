// src/layers/layers4d/mod.rs

use crate::tensor::Tensor4D;
use crate::model_plan::param_store::ParamSlice;

pub mod linear4d;
pub mod relu4d;
pub mod sigmoid4d;
pub mod softmax4d;
pub mod tanh4d;
pub mod memory4d;

pub use linear4d::Linear4D;
pub use relu4d::ReLU4D;
pub use sigmoid4d::Sigmoid4D;
pub use softmax4d::Softmax4D;
pub use tanh4d::Tanh4D;
pub use memory4d::Memory4D;

#[derive(Clone)]
pub enum LayerContext4D {
    Linear4D  { input: Tensor4D },
    ReLU4D    { input: Tensor4D },
    Sigmoid4D { output: Tensor4D },
    Tanh4D    { output: Tensor4D },
    Softmax4D { output: Tensor4D },
    Memory4D  { input: Tensor4D },
}

pub trait Layer4D: Send + Sync {
    fn input_dims(&self) -> Vec<usize>;
    fn output_dims(&self) -> Vec<usize>;

    fn forward(&self, inputs: &[Tensor4D], params: &[f32], slice: &ParamSlice) -> (Vec<Tensor4D>, Vec<LayerContext4D>) {
        let out_sizes = self.output_dims();
        let dim1 = inputs.first().map(|t| t.dim1).unwrap_or(0);
        let dim2 = inputs.first().map(|t| t.dim2).unwrap_or(0);
        let dim3 = inputs.first().map(|t| t.dim3).unwrap_or(0);
        let mut out_bufs: Vec<Vec<Vec<Vec<Vec<f32>>>>> = out_sizes.iter().map(|&sz| vec![vec![vec![vec![0.0; sz]; dim3]; dim2]; dim1]).collect();
        let ctxs = self.forward_into(inputs, params, slice, &mut out_bufs);
        let tensors = out_bufs.into_iter().map(Tensor4D::new).collect();
        (tensors, ctxs)
    }

    fn forward_into(
        &self,
        inputs: &[Tensor4D],
        params: &[f32],
        slice: &ParamSlice,
        out_bufs: &mut [Vec<Vec<Vec<Vec<f32>>>>],
    ) -> Vec<LayerContext4D>;

    fn backward(
        &self,
        ctxs: &[LayerContext4D],
        deltas: &[Tensor4D],
        params: &[f32],
        slice: &ParamSlice,
    ) -> (Vec<Tensor4D>, Vec<f32>);

    fn param_len(&self) -> usize;
}





