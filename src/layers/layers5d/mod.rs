use crate::tensor::Tensor5D;
use crate::jacobian::Jacobian5D;
use crate::model_plan::param_store::ParamSlice;

pub mod linear5d;
pub mod relu5d;
pub mod sigmoid5d;
pub mod softmax5d;
pub mod memory5d;
pub mod tanh5d;

pub use linear5d::Linear5D;
pub use relu5d::ReLU5D;
pub use sigmoid5d::Sigmoid5D;
pub use softmax5d::Softmax5D;
pub use memory5d::Memory5D;
pub use tanh5d::Tanh5D;

pub trait Layer5D {
    fn forward_5d(
        &self,
        input: &Tensor5D,
        j_input: &Jacobian5D,
        params: &[f32],
        slice: &ParamSlice,
    ) -> (Tensor5D, Jacobian5D);

    fn param_len(&self) -> usize;
}





