// src/layers/layers2d/mod.rs

use crate::tensor::Tensor2D;
use crate::model_plan::param_store::ParamSlice;

pub mod linear2d;
pub mod relu2d;
pub mod sigmoid2d;
pub mod softmax2d;
pub mod tanh2d;
pub mod memory2d;

pub use linear2d::Linear2D;
pub use relu2d::ReLU2D;
pub use sigmoid2d::Sigmoid2D;
pub use softmax2d::Softmax2D;
pub use tanh2d::Tanh2D;
pub use memory2d::Memory2D;

/// Контекст, сохраняемый 2D‑слоем во время прямого прохода.
#[derive(Clone)]
pub enum LayerContext {
    Linear2D  { input: Tensor2D },
    ReLU2D    { input: Tensor2D },
    Sigmoid2D { output: Tensor2D },
    Tanh2D    { output: Tensor2D },
    Softmax2D { output: Tensor2D },
    Memory2D  { input: Tensor2D },
}

pub trait Layer2D: Send + Sync {
    fn input_dims(&self) -> Vec<usize>;
    fn output_dims(&self) -> Vec<usize>;

    fn forward(&self, inputs: &[Tensor2D], params: &[f32], slice: &ParamSlice) -> (Vec<Tensor2D>, Vec<LayerContext>) {
        let out_sizes = self.output_dims();
        let dim1 = if let Some(first) = inputs.first() { first.dim1 } else { 0 };
        let mut out_bufs: Vec<Vec<Vec<f32>>> = out_sizes.iter().map(|&sz| vec![vec![0.0; sz]; dim1]).collect();
        let ctxs = self.forward_into(inputs, params, slice, &mut out_bufs);
        let tensors = out_bufs.into_iter().map(Tensor2D::new).collect();
        (tensors, ctxs)
    }

    fn forward_into(
        &self,
        inputs: &[Tensor2D],
        params: &[f32],
        slice: &ParamSlice,
        out_bufs: &mut [Vec<Vec<f32>>],
    ) -> Vec<LayerContext>;

    fn backward(
        &self,
        ctxs: &[LayerContext],
        deltas: &[Tensor2D],
        params: &[f32],
        slice: &ParamSlice,
    ) -> (Vec<Tensor2D>, Vec<f32>);

    fn param_len(&self) -> usize;
}





