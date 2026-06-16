use crate::tensor::Tensor3D;
use crate::jacobian::Jacobian3D;
use crate::model_plan::param_store::ParamSlice;

pub mod linear3d;
pub mod relu3d;
pub mod sigmoid3d;
pub mod softmax3d;
pub mod memory3d;
pub mod tanh3d;

pub use linear3d::Linear3D;
pub use relu3d::ReLU3D;
pub use sigmoid3d::Sigmoid3D;
pub use softmax3d::Softmax3D;
pub use memory3d::Memory3D;
pub use tanh3d::Tanh3D;

pub trait Layer3D {
    fn forward_3d(
        &self,
        input: &Tensor3D,
        j_input: &Jacobian3D,
        params: &[f32],
        slice: &ParamSlice,
    ) -> (Tensor3D, Jacobian3D);

    fn param_len(&self) -> usize;
}





