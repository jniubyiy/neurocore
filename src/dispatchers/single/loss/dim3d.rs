use crate::loss::ops::{LossInput, LossJacobian};
use crate::loss_plan::BuiltLoss;
use crate::dispatchers::common::model_trait::LossDispatch;

pub struct SingleLoss3D;

impl SingleLoss3D {
    pub fn new() -> Self { SingleLoss3D }
}

impl LossDispatch for SingleLoss3D {
    fn compute_loss(
        &self,
        pred: &dyn LossInput,
        target: &dyn LossInput,
        j_pred: &dyn LossJacobian,
        built_loss: &BuiltLoss,
    ) -> (f32, Vec<f32>) {
        (built_loss.forward)(pred, target, j_pred)
    }

    fn num_workers(&self) -> usize { 1 }
}