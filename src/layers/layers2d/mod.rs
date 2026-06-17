// ============================================================
// Файл: src/layers/layers2d/mod.rs
// ============================================================
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
    Linear2D     { input: Tensor2D },
    ReLU2D       { input: Tensor2D },
    Sigmoid2D    { output: Tensor2D },
    Tanh2D       { output: Tensor2D },
    Softmax2D    { output: Tensor2D },
    Sequential2D { contexts: Vec<LayerContext> },
}

impl LayerContext {
    /// Количество строк в батче для этого контекста.
    pub fn rows(&self) -> usize {
        match self {
            LayerContext::Linear2D { input } => input.rows,
            LayerContext::ReLU2D { input } => input.rows,
            LayerContext::Sigmoid2D { output } => output.rows,
            LayerContext::Tanh2D { output } => output.rows,
            LayerContext::Softmax2D { output } => output.rows,
            LayerContext::Sequential2D { contexts } => {
                contexts.first().map(|c| c.rows()).unwrap_or(0)
            }
        }
    }
}

pub trait Layer2D: Send + Sync {
    /// Обычный прямой проход (создаёт новый тензор) – для совместимости.
    fn forward(&self, input: &Tensor2D, params: &[f32], slice: &ParamSlice) -> (Tensor2D, LayerContext) {
        let rows = input.rows;
        let cols = self.out_features();
        let mut buf = vec![vec![0.0; cols]; rows];
        let ctx = self.forward_into(input, params, slice, &mut buf);
        (Tensor2D::new(buf), ctx)
    }

    /// Прямой проход с записью результата в предоставленный двумерный буфер.
    fn forward_into(&self, input: &Tensor2D, params: &[f32], slice: &ParamSlice, out_buf: &mut Vec<Vec<f32>>) -> LayerContext;

    fn backward(&self, ctx: &LayerContext, delta: &Tensor2D, params: &[f32], slice: &ParamSlice) -> (Tensor2D, Vec<f32>);

    fn param_len(&self) -> usize;
    fn in_features(&self) -> usize;
    fn out_features(&self) -> usize;
}





