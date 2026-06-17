use crate::tensor::Tensor3D;
use crate::loss_plan::BuiltLoss3D;
use crate::dispatchers::common::model_trait::LossDispatch3D;

pub struct SingleLoss3D;

impl SingleLoss3D {
    pub fn new() -> Self { SingleLoss3D }
}

impl LossDispatch3D for SingleLoss3D {
    fn compute_loss(&self, pred: &Tensor3D, target: &Tensor3D, built_loss: &BuiltLoss3D) -> (f32, Tensor3D) {
        (built_loss.forward)(pred, target)
    }

    fn num_workers(&self) -> usize { 1 }
}