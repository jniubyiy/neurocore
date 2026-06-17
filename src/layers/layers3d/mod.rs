// ============================================================
// Файл: src/layers/layers3d/mod.rs
// ============================================================
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

/// Контекст, сохраняемый 3D‑слоем во время прямого прохода.
#[derive(Clone)]
pub enum LayerContext3D {
    Linear3D   { contexts: Vec<super::layers2d::LayerContext> },
    ReLU3D     { input: Tensor3D },
    Sigmoid3D  { output: Tensor3D },
    Tanh3D     { output: Tensor3D },
    Softmax3D  { output: Tensor3D },
    Memory3D   { input: Tensor3D },
}

impl LayerContext3D {
    /// Количество 2D‑срезов (depth). Используется только вариантом `Linear3D`.
    pub fn depth(&self) -> usize {
        match self {
            LayerContext3D::Linear3D { contexts } => contexts.len(),
            _ => 0, // для активационных/Memory глубина не хранится списком контекстов
        }
    }

    /// Ссылка на массив 2D‑контекстов (только для `Linear3D`).
    pub fn contexts(&self) -> &Vec<super::layers2d::LayerContext> {
        match self {
            LayerContext3D::Linear3D { contexts } => contexts,
            _ => panic!("contexts() called on non‑Linear3D variant"),
        }
    }
}

pub trait Layer3D: Send + Sync {
    /// Обычный прямой проход (создаёт новый тензор) – для совместимости.
    fn forward(&self, input: &Tensor3D, params: &[f32], slice: &ParamSlice) -> (Tensor3D, LayerContext3D) {
        let depth = input.depth;
        let rows = input.rows;
        let cols = self.out_features();
        let mut buf = vec![vec![vec![0.0; cols]; rows]; depth];
        let ctx = self.forward_into(input, params, slice, &mut buf);
        (Tensor3D::new(buf), ctx)
    }

    /// Прямой проход с записью результата в предоставленный буфер.
    fn forward_into(&self, input: &Tensor3D, params: &[f32], slice: &ParamSlice, out_buf: &mut Vec<Vec<Vec<f32>>>) -> LayerContext3D;

    fn backward(&self, ctx: &LayerContext3D, delta: &Tensor3D, params: &[f32], slice: &ParamSlice) -> (Tensor3D, Vec<f32>);

    fn param_len(&self) -> usize;
    fn in_features(&self) -> usize;
    fn out_features(&self) -> usize;
}





