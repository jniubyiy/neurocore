use crate::tensor::{Tensor1D, Tensor2D, Tensor3D, Tensor4D, Tensor5D};
use crate::loss_plan::{BuiltLoss, BuiltLoss2D, BuiltLoss3D, BuiltLoss4D, BuiltLoss5D};
use crate::layers::LayerContext;
use crate::layers::layers3d::LayerContext3D;
use crate::layers::layers4d::LayerContext4D;
use crate::layers::layers5d::LayerContext5D;

pub trait Model1D {
    fn forward(&mut self, input: &Tensor1D) -> Tensor1D;
    fn backward(&mut self, delta: &Tensor1D) -> Tensor1D;
    fn update_params(&mut self, lr: f32);
    fn num_workers(&self) -> usize;
}

pub trait Model2D {
    fn forward(&self, input: &Tensor2D) -> (Tensor2D, Vec<Vec<LayerContext>>);
    fn backward(&self, contexts: &[Vec<LayerContext>], delta: &Tensor2D) -> (Tensor2D, Vec<Vec<f32>>);
    fn update_params(&mut self, lr: f32, all_grads: &[Vec<f32>]);
    fn num_workers(&self) -> usize;
}

pub trait Model3D {
    fn forward(&self, input: &Tensor3D) -> (Tensor3D, Vec<Vec<LayerContext3D>>);
    fn backward(&self, contexts: &[Vec<LayerContext3D>], delta: &Tensor3D) -> (Tensor3D, Vec<Vec<f32>>);
    fn update_params(&mut self, lr: f32, all_grads: &[Vec<f32>]);
    fn num_workers(&self) -> usize;
}

pub trait Model4D {
    fn forward(&self, input: &Tensor4D) -> (Tensor4D, Vec<Vec<LayerContext4D>>);
    fn backward(&self, contexts: &[Vec<LayerContext4D>], delta: &Tensor4D) -> (Tensor4D, Vec<Vec<f32>>);
    fn update_params(&mut self, lr: f32, all_grads: &[Vec<f32>]);
    fn num_workers(&self) -> usize;
}

pub trait Model5D {
    fn forward(&self, input: &Tensor5D) -> (Tensor5D, Vec<Vec<LayerContext5D>>);
    fn backward(&self, contexts: &[Vec<LayerContext5D>], delta: &Tensor5D) -> (Tensor5D, Vec<Vec<f32>>);
    fn update_params(&mut self, lr: f32, all_grads: &[Vec<f32>]);
    fn num_workers(&self) -> usize;
}

pub trait LossDispatch {
    fn compute_loss(
        &self,
        pred: &Tensor1D,
        target: &Tensor1D,
        built_loss: &BuiltLoss,
    ) -> (f32, Tensor1D);
    fn num_workers(&self) -> usize;
}

pub trait LossDispatch2D {
    fn compute_loss(
        &self,
        pred: &Tensor2D,
        target: &Tensor2D,
        built_loss: &BuiltLoss2D,
    ) -> (f32, Tensor2D);
    fn num_workers(&self) -> usize;
}

pub trait LossDispatch3D {
    fn compute_loss(
        &self,
        pred: &Tensor3D,
        target: &Tensor3D,
        built_loss: &BuiltLoss3D,
    ) -> (f32, Tensor3D);
    fn num_workers(&self) -> usize;
}

pub trait LossDispatch4D {
    fn compute_loss(
        &self,
        pred: &Tensor4D,
        target: &Tensor4D,
        built_loss: &BuiltLoss4D,
    ) -> (f32, Tensor4D);
    fn num_workers(&self) -> usize;
}

pub trait LossDispatch5D {
    fn compute_loss(
        &self,
        pred: &Tensor5D,
        target: &Tensor5D,
        built_loss: &BuiltLoss5D,
    ) -> (f32, Tensor5D);
    fn num_workers(&self) -> usize;
}