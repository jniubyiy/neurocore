// src/plans/model_plan/layer_desc.rs

use super::shape::Shape;
use super::blueprint::LayerKind;

#[derive(Debug, Clone)]
pub struct LayerDesc {
    pub name: String,
    pub kind: LayerKind,
    pub input_shape: Shape,
    pub output_shape: Shape,
    pub extra: Vec<f32>,   // дополнительные гиперпараметры (alpha, temperature и т.п.)
}

impl LayerDesc {
    pub fn new(kind: LayerKind) -> Self {
        Self {
            name: String::new(),
            kind,
            input_shape: Shape::single(0),
            output_shape: Shape::single(0),
            extra: Vec::new(),
        }
    }

    pub fn input(mut self, shape: Shape) -> Self {
        self.input_shape = shape;
        self
    }

    pub fn output(mut self, shape: Shape) -> Self {
        self.output_shape = shape;
        self
    }

    /// Устанавливает множественные входные формы (для Splitter, Combiner).
    /// Все потоки из переданных Shape объединяются в один Shape с несколькими потоками.
    pub fn inputs(mut self, shapes: Vec<Shape>) -> Self {
        let mut all_streams = Vec::new();
        let mut all_axes = Vec::new();
        for s in shapes {
            all_streams.extend(s.streams);
            all_axes.extend(s.axes);
        }
        self.input_shape = Shape {
            batch: 0,
            streams: all_streams,
            axes: all_axes,
        };
        self
    }

    /// Устанавливает множественные выходные формы.
    pub fn outputs(mut self, shapes: Vec<Shape>) -> Self {
        let mut all_streams = Vec::new();
        let mut all_axes = Vec::new();
        for s in shapes {
            all_streams.extend(s.streams);
            all_axes.extend(s.axes);
        }
        self.output_shape = Shape {
            batch: 0,
            streams: all_streams,
            axes: all_axes,
        };
        self
    }

    pub fn extra(mut self, values: Vec<f32>) -> Self {
        self.extra = values;
        self
    }

    /// Общее количество параметров слоя.
    pub fn param_len(&self) -> usize {
        match &self.kind {
            LayerKind::Linear => {
                assert_eq!(self.input_shape.streams.len(), 1,
                    "Linear layer expects exactly one input stream");
                assert_eq!(self.output_shape.streams.len(), 1,
                    "Linear layer expects exactly one output stream");
                let in_dim = self.input_shape.streams[0];
                let out_dim = self.output_shape.streams[0];
                in_dim * out_dim + out_dim
            }
            LayerKind::Combiner => {
                let streams = &self.input_shape.streams;
                assert_eq!(streams.len(), 2, "Combiner expects two input streams");
                let n = streams[0];
                let m = self.output_shape.streams[0];
                2 * m * n + m
            }
            LayerKind::Splitter => {
                let streams = &self.output_shape.streams;
                assert_eq!(streams.len(), 2, "Splitter expects two output streams");
                let n = self.input_shape.streams[0];
                let p = streams[0];
                let q = streams[1];
                p * n + q * n + p + q
            }
            LayerKind::Memory => 0,
            LayerKind::SoftSparseGate | LayerKind::SoftKeepGate => {
                assert_eq!(self.input_shape.streams.len(), 1,
                    "SoftSparseGate/SoftKeepGate expects one input stream");
                self.input_shape.streams[0]
            }
            LayerKind::DualAnchor => {
                assert_eq!(self.input_shape.streams.len(), 1,
                    "DualAnchor expects one input stream");
                2 * self.input_shape.streams[0] + 1
            }
            LayerKind::LeakyReLU | LayerKind::Identity => 0,
            LayerKind::AdaptivePerFeatureActivation => {
                assert_eq!(self.input_shape.streams.len(), 1,
                    "AdaptivePerFeatureActivation expects one input stream");
                let in_features = self.input_shape.streams[0];
                let num_activations = self.extra.get(0)
                    .map(|v| *v as usize)
                    .unwrap_or(4); // по умолчанию 4 базовые активации
                in_features * num_activations
            }
            LayerKind::SplitterConnector | LayerKind::CombinerConnector => 0,
            LayerKind::Unsqueeze | LayerKind::ReduceMean => 0,
            _ => 0,
        }
    }

    /// Создаёт универсальный слой по описанию.
    pub fn create_universal_layer(&self) -> Box<dyn crate::layers::UniversalLayer> {
        match self.kind {
            LayerKind::Linear => Box::new(crate::layers::Linear::new(
                self.input_shape.streams[0],
                self.output_shape.streams[0],
            )),
            LayerKind::ReLU => Box::new(crate::layers::ReLU::new()),
            LayerKind::Sigmoid => Box::new(crate::layers::Sigmoid::new()),
            LayerKind::Softmax => Box::new(crate::layers::Softmax::new()),
            LayerKind::Tanh => Box::new(crate::layers::Tanh::new()),
            LayerKind::Memory => Box::new(crate::layers::Memory::new(
                self.input_shape.streams[0],
                self.output_shape.streams[0],
            )),
            LayerKind::LeakyReLU => {
                let alpha = self.extra.get(0).copied().unwrap_or(0.01);
                Box::new(crate::layers::LeakyReLU::new(alpha))
            }
            LayerKind::Identity => Box::new(crate::layers::Identity::new()),
            LayerKind::SoftSparseGate => {
                let temp = self.extra.get(0).copied().unwrap_or(1.0);
                let features = self.input_shape.streams[0];
                Box::new(crate::layers::SoftSparseGate::new(features, temp))
            }
            LayerKind::SoftKeepGate => {
                let temp = self.extra.get(0).copied().unwrap_or(1.0);
                let features = self.input_shape.streams[0];
                Box::new(crate::layers::SoftKeepGate::new(features, temp))
            }
            LayerKind::DualAnchor => {
                let features = self.input_shape.streams[0];
                Box::new(crate::layers::DualAnchor::new(features, features))
            }
            LayerKind::AdaptivePerFeatureActivation => {
                let in_features = self.input_shape.streams[0];
                let num_activations = self.extra.get(0)
                    .map(|v| *v as usize)
                    .unwrap_or(4); // по умолчанию 4 базовые активации
                Box::new(crate::layers::AdaptivePerFeatureActivation::new(
                    in_features,
                    num_activations,
                ))
            }
            _ => panic!("Unsupported layer kind for UniversalLayer: {:?}", self.kind),
        }
    }
}