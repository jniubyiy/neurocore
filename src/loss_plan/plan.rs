use super::blueprint::LossBlueprint;
use crate::tensor::{Tensor1D, Tensor2D, Tensor3D, Tensor4D, Tensor5D};
use crate::loss::ops;
use std::sync::Arc;

pub struct BuiltLoss {
    pub forward: Arc<dyn Fn(&Tensor1D, &Tensor1D) -> (f32, Tensor1D) + Send + Sync>,
}

pub struct BuiltLoss2D {
    pub forward: Arc<dyn Fn(&Tensor2D, &Tensor2D) -> (f32, Tensor2D) + Send + Sync>,
}

pub struct BuiltLoss3D {
    pub forward: Arc<dyn Fn(&Tensor3D, &Tensor3D) -> (f32, Tensor3D) + Send + Sync>,
}

pub struct BuiltLoss4D {
    pub forward: Arc<dyn Fn(&Tensor4D, &Tensor4D) -> (f32, Tensor4D) + Send + Sync>,
}

pub struct BuiltLoss5D {
    pub forward: Arc<dyn Fn(&Tensor5D, &Tensor5D) -> (f32, Tensor5D) + Send + Sync>,
}

pub struct LossPlan {
    blueprint: LossBlueprint,
}

impl LossPlan {
    pub fn new(blueprint: LossBlueprint) -> Result<Self, String> {
        Ok(Self { blueprint })
    }

    pub fn build(&self) -> Result<BuiltLoss, String> {
        let forward: Box<dyn Fn(&Tensor1D, &Tensor1D) -> (f32, Tensor1D) + Send + Sync> = match &self.blueprint {
            LossBlueprint::MSE => Box::new(|p, t| ops::mse_loss(p, t)),
            LossBlueprint::MAE => Box::new(|p, t| ops::mae_loss(p, t)),
            LossBlueprint::CrossEntropy { .. } => Box::new(|p, t| ops::cross_entropy_loss(p, t)),
            _ => return Err("Unsupported loss type".into()),
        };
        Ok(BuiltLoss { forward: Arc::new(forward) })
    }

    pub fn build_2d(&self) -> Result<BuiltLoss2D, String> {
        let forward: Box<dyn Fn(&Tensor2D, &Tensor2D) -> (f32, Tensor2D) + Send + Sync> = match &self.blueprint {
            LossBlueprint::MSE => Box::new(|p, t| ops::mse_loss_2d(p, t)),
            LossBlueprint::MAE => Box::new(|p, t| ops::mae_loss_2d(p, t)),
            LossBlueprint::CrossEntropy { .. } => Box::new(|p, t| ops::cross_entropy_loss_2d(p, t)),
            _ => return Err("Unsupported loss type".into()),
        };
        Ok(BuiltLoss2D { forward: Arc::new(forward) })
    }

    pub fn build_3d(&self) -> Result<BuiltLoss3D, String> {
        let forward: Box<dyn Fn(&Tensor3D, &Tensor3D) -> (f32, Tensor3D) + Send + Sync> = match &self.blueprint {
            LossBlueprint::MSE => Box::new(|p, t| ops::mse_loss_3d(p, t)),
            LossBlueprint::MAE => Box::new(|p, t| ops::mae_loss_3d(p, t)),
            LossBlueprint::CrossEntropy { .. } => Box::new(|p, t| ops::cross_entropy_loss_3d(p, t)),
            _ => return Err("Unsupported loss type".into()),
        };
        Ok(BuiltLoss3D { forward: Arc::new(forward) })
    }

    pub fn build_4d(&self) -> Result<BuiltLoss4D, String> {
        let forward: Box<dyn Fn(&Tensor4D, &Tensor4D) -> (f32, Tensor4D) + Send + Sync> = match &self.blueprint {
            LossBlueprint::MSE => Box::new(|p, t| ops::mse_loss_4d(p, t)),
            LossBlueprint::MAE => Box::new(|p, t| ops::mae_loss_4d(p, t)),
            LossBlueprint::CrossEntropy { .. } => Box::new(|p, t| ops::cross_entropy_loss_4d(p, t)),
            _ => return Err("Unsupported loss type".into()),
        };
        Ok(BuiltLoss4D { forward: Arc::new(forward) })
    }

    pub fn build_5d(&self) -> Result<BuiltLoss5D, String> {
        let forward: Box<dyn Fn(&Tensor5D, &Tensor5D) -> (f32, Tensor5D) + Send + Sync> = match &self.blueprint {
            LossBlueprint::MSE => Box::new(|p, t| ops::mse_loss_5d(p, t)),
            LossBlueprint::MAE => Box::new(|p, t| ops::mae_loss_5d(p, t)),
            LossBlueprint::CrossEntropy { .. } => Box::new(|p, t| ops::cross_entropy_loss_5d(p, t)),
            _ => return Err("Unsupported loss type".into()),
        };
        Ok(BuiltLoss5D { forward: Arc::new(forward) })
    }
}