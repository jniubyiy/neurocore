use crate::tensor::{Tensor1D, Tensor2D, Tensor3D, Tensor4D, Tensor5D};
use crate::jacobian::{Jacobian, Jacobian2D, Jacobian3D, Jacobian4D, Jacobian5D};
use crate::loss::ops::{LossInput, LossJacobian};
use crate::loss_plan::BuiltLoss;

pub trait Model1D {
    fn forward(&mut self, input: &Tensor1D, j_input: &Jacobian) -> (Tensor1D, Jacobian);
    fn update_params(&mut self, lr: f32, grad: &[f32]);
    fn num_workers(&self) -> usize;
}

pub trait Model2D {
    fn forward(&mut self, input: &Tensor2D, j_input: &Jacobian2D) -> (Tensor2D, Jacobian2D);
    fn update_params(&mut self, lr: f32, grad: &[f32]);
    fn num_workers(&self) -> usize;
}

pub trait Model3D {
    fn forward(&mut self, input: &Tensor3D, j_input: &Jacobian3D) -> (Tensor3D, Jacobian3D);
    fn update_params(&mut self, lr: f32, grad: &[f32]);
    fn num_workers(&self) -> usize;
}

pub trait Model4D {
    fn forward(&mut self, input: &Tensor4D, j_input: &Jacobian4D) -> (Tensor4D, Jacobian4D);
    fn update_params(&mut self, lr: f32, grad: &[f32]);
    fn num_workers(&self) -> usize;
}

pub trait Model5D {
    fn forward(&mut self, input: &Tensor5D, j_input: &Jacobian5D) -> (Tensor5D, Jacobian5D);
    fn update_params(&mut self, lr: f32, grad: &[f32]);
    fn num_workers(&self) -> usize;
}

pub trait LossDispatch {
    fn compute_loss(
        &self,
        pred: &dyn LossInput,
        target: &dyn LossInput,
        j_pred: &dyn LossJacobian,
        built_loss: &BuiltLoss,
    ) -> (f32, Vec<f32>);
    fn num_workers(&self) -> usize;
}
