use crate::tensor::Tensor2D;
use crate::jacobian::Jacobian2D;
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

pub trait Layer2D {
    fn forward_2d(
        &self,
        input: &Tensor2D,
        j_input: &Jacobian2D,
        params: &[f32],
        slice: &ParamSlice,
    ) -> (Tensor2D, Jacobian2D);

    fn param_len(&self) -> usize;

    /// Размер входа (число признаков у каждого примера).
    fn in_features(&self) -> usize;
    /// Размер выхода (число выходных признаков).
    fn out_features(&self) -> usize;

    fn execute_range(
        &self,
        input: &Tensor2D,
        j_input: &Jacobian2D,
        out: &mut [f32],
        j_out: &mut [f32],
        row_start: usize,
        row_end: usize,
        col_start: usize,
        col_end: usize,
        total_params: usize,
        params: &[f32],
        slice: &ParamSlice,
    );
}





