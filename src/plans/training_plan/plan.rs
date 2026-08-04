// src/training_plan/plan.rs

use crate::loss_plan::desc::LossDesc;
use crate::model_plan::layer_desc::LayerDesc;
use crate::optimizer_plan::OptimizerDesc;
use crate::tensor::Tensor2D;
use super::profiling::ProfileMode;

#[derive(Debug, Clone)]
pub enum Initializer {
    Zeros,
    Ones,
    RandomUniform { min: f32, max: f32 },
}

#[derive(Debug, Clone)]
pub enum DataSource {
    Tensor2D(Tensor2D),
}

#[derive(Debug, Clone)]
pub struct ValidationConfig {
    pub data: DataSource,
    pub frequency: usize,
}

#[derive(Debug, Clone)]
pub struct TrainingPlan {
    pub model_fn: fn() -> Vec<LayerDesc>,
    pub loss_desc: LossDesc,
    pub optimizer_desc: OptimizerDesc,
    pub epochs: usize,
    pub batch_size: usize,
    pub train_data: Option<DataSource>,
    pub validation: Option<ValidationConfig>,
    pub test_data: Option<DataSource>,
    pub initializer: Initializer,
    pub seed: Option<u64>,
    pub output_tensors: Vec<String>,
    pub profile: ProfileMode,
}

impl TrainingPlan {
    pub fn new() -> Self {
        Self {
            model_fn: || panic!("Model function not set"),
            loss_desc: LossDesc::from_chain(
                crate::loss_plan::chain::ElementChain::new()
                    .add(Box::new(crate::loss_plan::cubes::Sub))
                    .add(Box::new(crate::loss_plan::cubes::Square)),
                crate::loss_plan::expr::Aggregation::Mean,
                0, 0, 0,
            ),
            optimizer_desc: OptimizerDesc::new()
                .add(crate::optimizer_plan::OptCubeDesc::ScaleGradient(0.01))
                .add(crate::optimizer_plan::OptCubeDesc::ApplyUpdate),
            epochs: 1,
            batch_size: 1,
            train_data: None,
            validation: None,
            test_data: None,
            initializer: Initializer::RandomUniform { min: -0.1, max: 0.1 },
            seed: None,
            output_tensors: Vec::new(),
            profile: ProfileMode::None,
        }
    }

    pub fn model(mut self, model_fn: fn() -> Vec<LayerDesc>) -> Self {
        self.model_fn = model_fn;
        self
    }
    pub fn loss(mut self, desc: LossDesc) -> Self {
        self.loss_desc = desc;
        self
    }
    pub fn optimizer(mut self, desc: OptimizerDesc) -> Self {
        self.optimizer_desc = desc;
        self
    }
    pub fn epochs(mut self, epochs: usize) -> Self {
        self.epochs = epochs;
        self
    }
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }
    pub fn train_data(mut self, data: DataSource) -> Self {
        self.train_data = Some(data);
        self
    }
    pub fn validation_data(mut self, data: DataSource, frequency: usize) -> Self {
        self.validation = Some(ValidationConfig { data, frequency });
        self
    }
    pub fn test_data(mut self, data: DataSource) -> Self {
        self.test_data = Some(data);
        self
    }
    pub fn init_weights(mut self, init: Initializer) -> Self {
        self.initializer = init;
        self
    }
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
    pub fn output_tensors(mut self, tensors: Vec<String>) -> Self {
        self.output_tensors = tensors;
        self
    }
    pub fn profile(mut self, mode: ProfileMode) -> Self {
        self.profile = mode;
        self
    }
}