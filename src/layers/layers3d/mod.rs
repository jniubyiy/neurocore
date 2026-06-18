// src/layers/layers3d/mod.rs

use crate::tensor::Tensor3D;
use crate::model_plan::param_store::ParamSlice;

pub mod linear3d;
pub mod relu3d;
pub mod sigmoid3d;
pub mod softmax3d;
pub mod tanh3d;
pub mod memory3d;

pub use linear3d::Linear3D;
pub use relu3d::ReLU3D;
pub use sigmoid3d::Sigmoid3D;
pub use softmax3d::Softmax3D;
pub use tanh3d::Tanh3D;
pub use memory3d::Memory3D;

#[derive(Clone)]
pub enum LayerContext3D {
    Linear3D  { input: Tensor3D },
    ReLU3D    { input: Tensor3D },
    Sigmoid3D { output: Tensor3D },
    Tanh3D    { output: Tensor3D },
    Softmax3D { output: Tensor3D },
    Memory3D  { input: Tensor3D },
}

pub trait Layer3D: Send + Sync {
    fn input_dims(&self) -> Vec<usize>;
    fn output_dims(&self) -> Vec<usize>;

    fn forward(&self, inputs: &[Tensor3D], params: &[f32], slice: &ParamSlice) -> (Vec<Tensor3D>, Vec<LayerContext3D>) {
        let out_sizes = self.output_dims();
        let dim1 = inputs.first().map(|t| t.dim1).unwrap_or(0);
        let dim2 = inputs.first().map(|t| t.dim2).unwrap_or(0);
        let mut out_bufs: Vec<Vec<Vec<Vec<f32>>>> = out_sizes.iter().map(|&sz| vec![vec![vec![0.0; sz]; dim2]; dim1]).collect();
        let ctxs = self.forward_into(inputs, params, slice, &mut out_bufs);
        let tensors = out_bufs.into_iter().map(Tensor3D::new).collect();
        (tensors, ctxs)
    }

    fn forward_into(
        &self,
        inputs: &[Tensor3D],
        params: &[f32],
        slice: &ParamSlice,
        out_bufs: &mut [Vec<Vec<Vec<f32>>>],
    ) -> Vec<LayerContext3D>;

    fn backward(
        &self,
        ctxs: &[LayerContext3D],
        deltas: &[Tensor3D],
        params: &[f32],
        slice: &ParamSlice,
    ) -> (Vec<Tensor3D>, Vec<f32>);

    fn param_len(&self) -> usize;
}





