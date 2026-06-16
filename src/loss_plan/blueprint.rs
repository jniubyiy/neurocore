#[derive(Debug, Clone)]
pub enum LossBlueprint {
    MSE,
    MAE,
    CrossEntropy { num_classes: usize },
    Sum(Vec<LossBlueprint>),
    Scale(f32, Box<LossBlueprint>),
}

impl LossBlueprint {
    pub fn mse() -> Self { Self::MSE }
    pub fn mae() -> Self { Self::MAE }
    pub fn cross_entropy(num_classes: usize) -> Self { Self::CrossEntropy { num_classes } }
    pub fn sum(losses: Vec<LossBlueprint>) -> Self { Self::Sum(losses) }
    pub fn scale(self, factor: f32) -> Self { Self::Scale(factor, Box::new(self)) }
}