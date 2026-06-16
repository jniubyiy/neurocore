pub mod model;
pub mod loss;

pub use model::{TrainedModel1D, TrainedModel2D, TrainedModel3D, TrainedModel4D, TrainedModel5D};
pub use loss::{TrainedLoss1D, TrainedLoss2D, TrainedLoss3D, TrainedLoss4D, TrainedLoss5D};