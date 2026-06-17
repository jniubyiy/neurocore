// ============================================================
// Файл: src/layers/layers4d/mod.rs
// ============================================================
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
    Linear4D   { contexts: Vec<super::layers3d::LayerContext3D> },
    ReLU4D     { input: Tensor4D },
    Sigmoid4D  { output: Tensor4D },
    Tanh4D     { output: Tensor4D },
    Softmax4D  { output: Tensor4D },
    Memory4D   { input: Tensor4D },
}

impl LayerContext4D {
    pub fn dim1(&self) -> usize {
        match self {
            LayerContext4D::Linear4D { contexts } => contexts.len(),
            _ => 0,
        }
    }

    pub fn contexts(&self) -> &Vec<super::layers3d::LayerContext3D> {
        match self {
            LayerContext4D::Linear4D { contexts } => contexts,
            _ => panic!("contexts() called on non‑Linear4D variant"),
        }
    }
}

pub trait Layer4D: Send + Sync {
    fn forward(&self, input: &Tensor4D, params: &[f32], slice: &ParamSlice) -> (Tensor4D, LayerContext4D) {
        let dim1 = input.dim1;
        let depth = input.depth;
        let rows = input.rows;
        let cols = self.out_features();
        let mut buf = vec![vec![vec![vec![0.0; cols]; rows]; depth]; dim1];
        let ctx = self.forward_into(input, params, slice, &mut buf);
        (Tensor4D::new(buf), ctx)
    }

    fn forward_into(&self, input: &Tensor4D, params: &[f32], slice: &ParamSlice, out_buf: &mut Vec<Vec<Vec<Vec<f32>>>>) -> LayerContext4D;

    fn backward(&self, ctx: &LayerContext4D, delta: &Tensor4D, params: &[f32], slice: &ParamSlice) -> (Tensor4D, Vec<f32>);

    fn param_len(&self) -> usize;
    fn in_features(&self) -> usize;
    fn out_features(&self) -> usize;
}





