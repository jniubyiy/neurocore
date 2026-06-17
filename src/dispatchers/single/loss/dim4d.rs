use crate::tensor::Tensor4D;
use crate::loss_plan::BuiltLoss4D;
use crate::dispatchers::common::model_trait::LossDispatch4D;

pub struct SingleLoss4D;

impl SingleLoss4D {
    pub fn new() -> Self { SingleLoss4D }
}

impl LossDispatch4D for SingleLoss4D {
    fn compute_loss(&self, pred: &Tensor4D, target: &Tensor4D, built_loss: &BuiltLoss4D) -> (f32, Tensor4D) {
        (built_loss.forward)(pred, target)
    }

    fn num_workers(&self) -> usize { 1 }
}