pub mod model;
pub mod loss;

pub use model::{SingleModel1D, SingleModel2D, SingleModel3D, SingleModel4D, SingleModel5D};
pub use loss::{SingleLoss1D, SingleLoss2D, SingleLoss3D, SingleLoss4D, SingleLoss5D};