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

    pub fn build(&self, _pred_cols: usize, _target_cols: usize) -> Result<BuiltLoss, String> {
        let forward = match &self.blueprint {
            LossBlueprint::MSE => {
                Box::new(|pred: &Tensor1D, target: &Tensor1D| ops::mse_loss(pred, target))
                    as Box<dyn Fn(&Tensor1D, &Tensor1D) -> (f32, Tensor1D) + Send + Sync>
            }
            LossBlueprint::MAE => {
                Box::new(|pred: &Tensor1D, target: &Tensor1D| ops::mae_loss(pred, target))
            }
            LossBlueprint::CrossEntropy { .. } => {
                Box::new(|pred: &Tensor1D, target: &Tensor1D| ops::cross_entropy_loss(pred, target))
            }
            _ => return Err("Unsupported loss type".into()),
        };
        Ok(BuiltLoss { forward: Arc::new(forward) })
    }

    pub fn build_2d(&self, _pred_cols: usize, _target_cols: usize) -> Result<BuiltLoss2D, String> {
        let forward = match &self.blueprint {
            LossBlueprint::MSE => {
                Box::new(|pred: &Tensor2D, target: &Tensor2D| ops::mse_loss_2d(pred, target))
                    as Box<dyn Fn(&Tensor2D, &Tensor2D) -> (f32, Tensor2D) + Send + Sync>
            }
            LossBlueprint::MAE => {
                Box::new(|pred: &Tensor2D, target: &Tensor2D| ops::mae_loss_2d(pred, target))
            }
            LossBlueprint::CrossEntropy { .. } => {
                Box::new(|pred: &Tensor2D, target: &Tensor2D| ops::cross_entropy_loss_2d(pred, target))
            }
            _ => return Err("Unsupported loss type".into()),
        };
        Ok(BuiltLoss2D { forward: Arc::new(forward) })
    }

    pub fn build_3d(&self, _pred_cols: usize, _target_cols: usize) -> Result<BuiltLoss3D, String> {
        let forward = match &self.blueprint {
            LossBlueprint::MSE => {
                Box::new(|pred: &Tensor3D, target: &Tensor3D| ops::mse_loss_3d(pred, target))
                    as Box<dyn Fn(&Tensor3D, &Tensor3D) -> (f32, Tensor3D) + Send + Sync>
            }
            LossBlueprint::MAE => {
                Box::new(|pred: &Tensor3D, target: &Tensor3D| ops::mae_loss_3d(pred, target))
            }
            LossBlueprint::CrossEntropy { .. } => {
                Box::new(|pred: &Tensor3D, target: &Tensor3D| ops::cross_entropy_loss_3d(pred, target))
            }
            _ => return Err("Unsupported loss type".into()),
        };
        Ok(BuiltLoss3D { forward: Arc::new(forward) })
    }

    pub fn build_4d(&self, _pred_cols: usize, _target_cols: usize) -> Result<BuiltLoss4D, String> {
        let forward = match &self.blueprint {
            LossBlueprint::MSE => {
                Box::new(|pred: &Tensor4D, target: &Tensor4D| ops::mse_loss_4d(pred, target))
                    as Box<dyn Fn(&Tensor4D, &Tensor4D) -> (f32, Tensor4D) + Send + Sync>
            }
            LossBlueprint::MAE => {
                Box::new(|pred: &Tensor4D, target: &Tensor4D| ops::mae_loss_4d(pred, target))
            }
            LossBlueprint::CrossEntropy { .. } => {
                Box::new(|pred: &Tensor4D, target: &Tensor4D| ops::cross_entropy_loss_4d(pred, target))
            }
            _ => return Err("Unsupported loss type".into()),
        };
        Ok(BuiltLoss4D { forward: Arc::new(forward) })
    }

    pub fn build_5d(&self, _pred_cols: usize, _target_cols: usize) -> Result<BuiltLoss5D, String> {
        let forward = match &self.blueprint {
            LossBlueprint::MSE => {
                Box::new(|pred: &Tensor5D, target: &Tensor5D| ops::mse_loss_5d(pred, target))
                    as Box<dyn Fn(&Tensor5D, &Tensor5D) -> (f32, Tensor5D) + Send + Sync>
            }
            LossBlueprint::MAE => {
                Box::new(|pred: &Tensor5D, target: &Tensor5D| ops::mae_loss_5d(pred, target))
            }
            LossBlueprint::CrossEntropy { .. } => {
                Box::new(|pred: &Tensor5D, target: &Tensor5D| ops::cross_entropy_loss_5d(pred, target))
            }
            _ => return Err("Unsupported loss type".into()),
        };
        Ok(BuiltLoss5D { forward: Arc::new(forward) })
    }
}