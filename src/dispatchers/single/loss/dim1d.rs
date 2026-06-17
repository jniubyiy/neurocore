use crate::tensor::Tensor1D;
use crate::loss_plan::BuiltLoss;
use crate::dispatchers::common::model_trait::LossDispatch;

pub struct SingleLoss1D;

impl SingleLoss1D {
    pub fn new() -> Self { SingleLoss1D }
}

impl LossDispatch for SingleLoss1D {
    fn compute_loss(
        &self,
        pred: &Tensor1D,
        target: &Tensor1D,
        built_loss: &BuiltLoss,
    ) -> (f32, Tensor1D) {
        (built_loss.forward)(pred, target)
    }

    fn num_workers(&self) -> usize { 1 }
}