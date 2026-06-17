// ============================================================
// Файл: src/layers/layers5d/mod.rs
// ============================================================
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
    Linear5D   { contexts: Vec<super::layers4d::LayerContext4D> },
    ReLU5D     { input: Tensor5D },
    Sigmoid5D  { output: Tensor5D },
    Tanh5D     { output: Tensor5D },
    Softmax5D  { output: Tensor5D },
    Memory5D   { input: Tensor5D },
}

impl LayerContext5D {
    pub fn outer(&self) -> usize {
        match self {
            LayerContext5D::Linear5D { contexts } => contexts.len(),
            _ => 0,
        }
    }

    pub fn contexts(&self) -> &Vec<super::layers4d::LayerContext4D> {
        match self {
            LayerContext5D::Linear5D { contexts } => contexts,
            _ => panic!("contexts() called on non‑Linear5D variant"),
        }
    }
}

pub trait Layer5D: Send + Sync {
    fn forward(&self, input: &Tensor5D, params: &[f32], slice: &ParamSlice) -> (Tensor5D, LayerContext5D) {
        let outer = input.outer;
        let dim1 = input.dim1;
        let depth = input.depth;
        let rows = input.rows;
        let cols = self.out_features();
        let mut buf = vec![vec![vec![vec![vec![0.0; cols]; rows]; depth]; dim1]; outer];
        let ctx = self.forward_into(input, params, slice, &mut buf);
        (Tensor5D::new(buf), ctx)
    }

    fn forward_into(&self, input: &Tensor5D, params: &[f32], slice: &ParamSlice, out_buf: &mut Vec<Vec<Vec<Vec<Vec<f32>>>>>) -> LayerContext5D;

    fn backward(&self, ctx: &LayerContext5D, delta: &Tensor5D, params: &[f32], slice: &ParamSlice) -> (Tensor5D, Vec<f32>);

    fn param_len(&self) -> usize;
    fn in_features(&self) -> usize;
    fn out_features(&self) -> usize;
}





