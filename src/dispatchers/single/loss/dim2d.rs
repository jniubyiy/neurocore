use crate::tensor::Tensor2D;
use crate::loss_plan::BuiltLoss2D;
use crate::dispatchers::common::model_trait::LossDispatch2D; // определим ниже

pub struct SingleLoss2D;

impl SingleLoss2D {
    pub fn new() -> Self { SingleLoss2D }
}

impl LossDispatch2D for SingleLoss2D {
    fn compute_loss(
        &self,
        pred: &Tensor2D,
        target: &Tensor2D,
        built_loss: &BuiltLoss2D,
    ) -> (f32, Tensor2D) {
        (built_loss.forward)(pred, target)
    }

    fn num_workers(&self) -> usize { 1 }
}