// src/training_plan/plan.rs

use crate::loss_plan::desc::LossDesc;
use crate::model_plan::layer_desc::LayerDesc;
use crate::optimizer_plan::OptimizerDesc;
use crate::tensor::{Tensor2D, Tensor3D, Tensor4D, Tensor5D};
use crate::logging::training_monitor::MonitorConfig;
use crate::device_plan::DevicePlan;
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
    Tensor3D(Tensor3D),
    Tensor4D(Tensor4D),
    Tensor5D(Tensor5D),
}

impl DataSource {
    /// Создаёт DataSource из Tensor2D.
    pub fn from_tensor2d(tensor: Tensor2D) -> Self {
        DataSource::Tensor2D(tensor)
    }

    /// Создаёт DataSource из Tensor3D.
    pub fn from_tensor3d(tensor: Tensor3D) -> Self {
        DataSource::Tensor3D(tensor)
    }

    /// Создаёт DataSource из Tensor4D.
    pub fn from_tensor4d(tensor: Tensor4D) -> Self {
        DataSource::Tensor4D(tensor)
    }

    /// Создаёт DataSource из Tensor5D.
    pub fn from_tensor5d(tensor: Tensor5D) -> Self {
        DataSource::Tensor5D(tensor)
    }
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
    /// Входные данные для обучения (один поток). Для многопотоковых моделей
    /// используйте `train_data_streams`.
    pub train_data: Option<DataSource>,
    /// Если задана, используется как цель при вычислении потерь.
    /// Если не задана, цель берётся из `train_data` (автоэнкодер).
    pub target_data: Option<DataSource>,
    pub validation: Option<ValidationConfig>,
    /// Входные данные для финального тестирования (один поток).
    /// Если не задано, тестирование не производится.
    pub test_data: Option<DataSource>,
    /// Отдельная цель для финального тестирования (один поток).
    /// Если задано `test_data`, но `test_target_data` не задано,
    /// то цель при тестировании совпадает с входом (режим автоэнкодера).
    pub test_target_data: Option<DataSource>,
    pub initializer: Initializer,
    pub seed: Option<u64>,
    pub output_tensors: Vec<String>,
    pub profile: ProfileMode,
    pub monitoring: bool,
    pub monitor_config: MonitorConfig,

    // Задел на многопотоковость: эти поля позволяют задать отдельные DataSource
    // для каждого входного/выходного потока модели (Splitter/Combiner и т.п.).
    // Пока не используются в стандартном процессе обучения, но доступны для
    // ручного вызова forward_multi/backward_multi и будут интегрированы позже.
    pub train_data_streams: Option<Vec<DataSource>>,
    pub target_data_streams: Option<Vec<DataSource>>,
    pub test_input_streams: Option<Vec<DataSource>>,
    pub test_target_streams: Option<Vec<DataSource>>,
}

impl TrainingPlan {
    pub fn new() -> Self {
        Self {
            model_fn: || panic!("Model function not set"),
            loss_desc: LossDesc::from_chain(
                crate::loss_plan::chain::ElementChain::new()
                    .add(Box::new(crate::loss_plan::cubes::Sub::new(1)))
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
            target_data: None,
            validation: None,
            test_data: None,
            test_target_data: None,
            initializer: Initializer::RandomUniform { min: -0.1, max: 0.1 },
            seed: None,
            output_tensors: Vec::new(),
            profile: ProfileMode::None,
            monitoring: false,
            monitor_config: MonitorConfig::default(),
            train_data_streams: None,
            target_data_streams: None,
            test_input_streams: None,
            test_target_streams: None,
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
    /// Устанавливает отдельные целевые данные. Если не задано, цель = train_data.
    pub fn target_data(mut self, data: DataSource) -> Self {
        self.target_data = Some(data);
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
    /// Устанавливает отдельные целевые данные для финального тестирования.
    /// Если задано `test_data`, но `test_target_data` не задано,
    /// то цель при тестировании совпадает с входом (режим автоэнкодера).
    pub fn test_target_data(mut self, data: DataSource) -> Self {
        self.test_target_data = Some(data);
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

    /// Включает мониторинг обучения с параметрами по умолчанию.
    pub fn enable_monitoring(mut self) -> Self {
        self.monitoring = true;
        self
    }

    /// Устанавливает пользовательскую конфигурацию мониторинга.
    pub fn with_monitor_config(mut self, config: MonitorConfig) -> Self {
        self.monitoring = true;
        self.monitor_config = config;
        self
    }

    /// Строит модель по текущему плану, используя переданный план устройств.
    /// Это единственный публичный способ получить модель.
    pub fn build_model(&self, device_plan: DevicePlan) -> crate::compute_manager::graph::model::MixedModel {
        let model_desc = (self.model_fn)();
        crate::compute_manager::graph::model::MixedModel::from_plan_with_device_plan(
            model_desc,
            device_plan,
        )
        .expect("Failed to build model from training plan")
    }

    // ── Методы для множественных потоков (задел на будущее) ──

    /// Устанавливает несколько входных DataSource для обучения
    /// (по одному на каждый входной поток модели).
    /// При использовании этого метода `train_data` игнорируется.
    pub fn train_data_multi(mut self, streams: Vec<DataSource>) -> Self {
        self.train_data_streams = Some(streams);
        self
    }

    /// Устанавливает несколько целевых DataSource для обучения
    /// (по одному на каждый выходной поток модели).
    /// При использовании этого метода `target_data` игнорируется.
    pub fn target_data_multi(mut self, streams: Vec<DataSource>) -> Self {
        self.target_data_streams = Some(streams);
        self
    }

    /// Устанавливает несколько входных DataSource для финального тестирования.
    /// При использовании этого метода `test_data` игнорируется.
    pub fn test_input_multi(mut self, streams: Vec<DataSource>) -> Self {
        self.test_input_streams = Some(streams);
        self
    }

    /// Устанавливает несколько целевых DataSource для финального тестирования.
    /// При использовании этого метода `test_target_data` игнорируется.
    pub fn test_target_multi(mut self, streams: Vec<DataSource>) -> Self {
        self.test_target_streams = Some(streams);
        self
    }
}