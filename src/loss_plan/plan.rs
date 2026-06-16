use super::blueprint::LossBlueprint;
use crate::loss::ops::{self, LossInput, LossJacobian};
use crate::model_plan::blueprint::is_power_of_two;
use std::sync::Arc;

pub struct BuiltLoss {
    pub forward: Arc<dyn Fn(&dyn LossInput, &dyn LossInput, &dyn LossJacobian) -> (f32, Vec<f32>) + Send + Sync>,
}

pub struct LossPlan {
    blueprint: LossBlueprint,
}

impl LossPlan {
    pub fn new(blueprint: LossBlueprint) -> Result<Self, String> {
        Self::validate(&blueprint)?;
        Ok(Self { blueprint })
    }

    fn validate(bp: &LossBlueprint) -> Result<(), String> {
        match bp {
            LossBlueprint::CrossEntropy { num_classes } if *num_classes < 2 =>
                Err("CrossEntropy требует минимум 2 класса.".into()),
            LossBlueprint::Sum(parts) => {
                if parts.is_empty() { return Err("Сумма потерь не может быть пустой.".into()); }
                for p in parts { Self::validate(p)?; }
                Ok(())
            },
            LossBlueprint::Scale(_, inner) => Self::validate(inner),
            _ => Ok(()),
        }
    }

    pub fn build(&self, pred_cols: usize, target_cols: usize) -> Result<BuiltLoss, String> {
        self.check_dims(&self.blueprint, pred_cols, target_cols)?;
        let forward = self.compile(&self.blueprint);
        Ok(BuiltLoss { forward: Arc::new(forward) })
    }

    fn check_dims(&self, bp: &LossBlueprint, pred_cols: usize, target_cols: usize) -> Result<(), String> {
        match bp {
            LossBlueprint::MSE | LossBlueprint::MAE => {
                if !is_power_of_two(pred_cols) {
                    return Err(format!("{}: pred_cols ({}) должен быть степенью двойки",
                        if matches!(bp, LossBlueprint::MSE) { "MSE" } else { "MAE" }, pred_cols));
                }
                if !is_power_of_two(target_cols) {
                    return Err(format!("{}: target_cols ({}) должен быть степенью двойки",
                        if matches!(bp, LossBlueprint::MSE) { "MSE" } else { "MAE" }, target_cols));
                }
                if pred_cols != target_cols {
                    return Err(format!("{}: pred_cols ({}) != target_cols ({})",
                        if matches!(bp, LossBlueprint::MSE) { "MSE" } else { "MAE" }, pred_cols, target_cols));
                }
            }
            LossBlueprint::CrossEntropy { num_classes } => {
                if !is_power_of_two(*num_classes) {
                    return Err(format!("CrossEntropy: num_classes ({}) должен быть степенью двойки", num_classes));
                }
                if pred_cols != *num_classes {
                    return Err(format!("CrossEntropy: pred_cols ({}) != num_classes ({})", pred_cols, num_classes));
                }
                if target_cols != 1 {
                    return Err("CrossEntropy: target_cols должен быть 1".into());
                }
            }
            LossBlueprint::Sum(parts) => {
                for p in parts { self.check_dims(p, pred_cols, target_cols)?; }
            }
            LossBlueprint::Scale(_, inner) => self.check_dims(inner, pred_cols, target_cols)?,
        }
        Ok(())
    }

    fn compile(&self, bp: &LossBlueprint) -> Box<dyn Fn(&dyn LossInput, &dyn LossInput, &dyn LossJacobian) -> (f32, Vec<f32>) + Send + Sync> {
        match bp {
            LossBlueprint::MSE => Box::new(|pred, target, j_pred| {
                let (diff, j_diff) = ops::sub(pred, target, j_pred);
                let (sq, j_sq) = ops::square(&*diff, &*j_diff);
                ops::mean(&*sq, &*j_sq)
            }),
            LossBlueprint::MAE => Box::new(|pred, target, j_pred| {
                let (diff, j_diff) = ops::sub(pred, target, j_pred);
                let (abs, j_abs) = ops::abs(&*diff, &*j_diff);
                ops::mean(&*abs, &*j_abs)
            }),
            LossBlueprint::CrossEntropy { .. } => Box::new(|logits, target, j_logits| {
                let (soft, j_soft) = ops::softmax(logits, j_logits);
                let (log_soft, j_log_soft) = ops::log(&*soft, &*j_soft);
                ops::gather_neg_mean(&*log_soft, &*j_log_soft, target)
            }),
            LossBlueprint::Sum(parts) => {
                let compiled: Vec<_> = parts.iter().map(|p| self.compile(p)).collect();
                Box::new(move |pred, target, j| {
                    let mut total_loss = 0.0;
                    let mut total_grad = vec![0.0; j.params()];
                    for f in &compiled {
                        let (loss, grad) = f(pred, target, j);
                        total_loss += loss;
                        for (i, g) in grad.iter().enumerate() { total_grad[i] += g; }
                    }
                    (total_loss, total_grad)
                })
            },
            LossBlueprint::Scale(factor, inner) => {
                let factor = *factor;
                let compiled = self.compile(inner);
                Box::new(move |pred, target, j| {
                    let (loss, grad) = compiled(pred, target, j);
                    (loss * factor, grad.iter().map(|g| g * factor).collect())
                })
            },
        }
    }
}