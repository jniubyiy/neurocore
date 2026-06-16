use crate::tensor::Tensor4D;
use crate::jacobian::Jacobian4D;
use crate::model_plan::param_store::ParamSlice;

pub mod linear4d;
pub mod relu4d;
pub mod sigmoid4d;
pub mod softmax4d;
pub mod memory4d;
pub mod tanh4d;

pub use linear4d::Linear4D;
pub use relu4d::ReLU4D;
pub use sigmoid4d::Sigmoid4D;
pub use softmax4d::Softmax4D;
pub use memory4d::Memory4D;
pub use tanh4d::Tanh4D;

pub trait Layer4D {
    fn forward_4d(
        &self,
        input: &Tensor4D,
        j_input: &Jacobian4D,
        params: &[f32],
        slice: &ParamSlice,
    ) -> (Tensor4D, Jacobian4D);

    fn param_len(&self) -> usize;
}





