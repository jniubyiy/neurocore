use crate::tensor::Tensor5D;
use crate::loss_plan::BuiltLoss5D;
use crate::dispatchers::common::model_trait::LossDispatch5D;

pub struct SingleLoss5D;

impl SingleLoss5D {
    pub fn new() -> Self { SingleLoss5D }
}

impl LossDispatch5D for SingleLoss5D {
    fn compute_loss(&self, pred: &Tensor5D, target: &Tensor5D, built_loss: &BuiltLoss5D) -> (f32, Tensor5D) {
        (built_loss.forward)(pred, target)
    }
    fn num_workers(&self) -> usize { 1 }
}