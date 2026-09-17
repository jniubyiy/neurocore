// src/plans/model_plan/layer_desc.rs

use super::shape::Shape;
use super::blueprint::LayerKind;

#[derive(Debug, Clone)]
pub struct LayerDesc {
    pub name: String,
    pub kind: LayerKind,
    pub input_shape: Shape,
    pub output_shape: Shape,
    pub extra: Vec<f32>,
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
                    .unwrap_or(4);
                in_features * num_activations
            }
            LayerKind::SplitterConnector | LayerKind::CombinerConnector => 0,
            LayerKind::Unsqueeze | LayerKind::ReduceMean => 0,

            LayerKind::DualSlopeReLU => {
                assert_eq!(self.input_shape.streams.len(), 1,
                    "DualSlopeReLU expects one input stream");
                2 * self.input_shape.streams[0]
            }
            LayerKind::LearnableMish => {
                assert_eq!(self.input_shape.streams.len(), 1,
                    "LearnableMish expects one input stream");
                1
            }
            LayerKind::LearnableSoftplus => {
                assert_eq!(self.input_shape.streams.len(), 1,
                    "LearnableSoftplus expects one input stream");
                2 * self.input_shape.streams[0]
            }
            LayerKind::AdaptiveNormalization => {
                assert_eq!(self.input_shape.streams.len(), 1,
                    "AdaptiveNormalization expects one input stream");
                7 * self.input_shape.streams[0]
            }
            LayerKind::BatchRenorm1d => {
                assert_eq!(self.input_shape.streams.len(), 1,
                    "BatchRenorm1d expects one input stream");
                4 * self.input_shape.streams[0]
            }
            LayerKind::RMSNormWithLearnableEpsilon => {
                assert_eq!(self.input_shape.streams.len(), 1,
                    "RMSNormWithLearnableEpsilon expects one input stream");
                2 * self.input_shape.streams[0]
            }
            LayerKind::ConcreteDropout => 1,
            LayerKind::AdaptiveDropout => {
                assert_eq!(self.input_shape.streams.len(), 1,
                    "AdaptiveDropout expects one input stream");
                2 * self.input_shape.streams[0]
            }
            LayerKind::LinearAttention => {
                let d_model = self.extra.get(1).copied().unwrap_or(1.0) as usize;
                let min_heads = self.extra.get(2).copied().unwrap_or(1.0).max(1.0) as usize;
                let max_heads_raw = self.extra.get(3).copied().unwrap_or(min_heads as f32) as usize;
                let max_heads = max_heads_raw.max(min_heads);
                assert!(
                    max_heads > 0 && d_model % max_heads == 0,
                    "LinearAttention: d_model ({}) must be divisible by max_heads ({}).",
                    d_model,
                    max_heads
                );
                let d_head = d_model / max_heads;
                let head_param_count = 4 * d_head * d_model + 3 * d_head + d_model + 1;
                max_heads * head_param_count + 1
            }
            LayerKind::RelativePositionAttention => {
                let seq_len = self.extra.get(0).copied().unwrap_or(1.0) as usize;
                let d_model = self.extra.get(1).copied().unwrap_or(1.0) as usize;
                4 * (d_model * d_model + d_model) + (2 * seq_len - 1)
            }
            LayerKind::IndRNN => {
                let input_dim = self.extra.get(0).copied().unwrap_or(1.0) as usize;
                let _seq_len = self.extra.get(1).copied().unwrap_or(1.0) as usize;
                input_dim * input_dim + 2 * input_dim
            }
            LayerKind::Mamba => {
                let _seq_len = self.extra.get(0).copied().unwrap_or(1.0) as usize;
                let input_dim = self.extra.get(1).copied().unwrap_or(1.0) as usize;
                let state_dim = self.extra.get(2).copied().unwrap_or(1.0) as usize;
                state_dim * state_dim + state_dim * input_dim + input_dim * state_dim + 2
            }
            LayerKind::FeatureFusion => {
                let in_features = self.input_shape.streams[0];
                let out_features = self.output_shape.streams[0];
                out_features * (in_features + 1)
            }
            LayerKind::SpectrallyNormalizedLinear => {
                let in_features = self.input_shape.streams[0];
                let out_features = self.output_shape.streams[0];
                in_features * out_features + out_features + 1
            }
            LayerKind::SparseFeatureSelectionGate => {
                let features = self.input_shape.streams[0];
                features + 1
            }
            LayerKind::MultiResolutionKANLinear => {
                let in_features = self.input_shape.streams[0];
                let out_features = self.output_shape.streams[0];
                const G_C: usize = 3;
                const G_F: usize = 8;
                const K: usize = 3;
                out_features + in_features * out_features * (4 + (G_C + K) + (G_F + K))
            }
            LayerKind::PerFeatureAttention => {
                // extra = [seq_len, d_model]; d_head фиксирован в слое.
                let d_model = self.extra.get(1).copied().unwrap_or(1.0) as usize;
                let d_head = crate::layers::PerFeatureAttention::DEFAULT_D_HEAD;
                d_model * (10 * d_head + 2)
            }
            _ => 0,
        }
    }

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
                    .unwrap_or(4);
                Box::new(crate::layers::AdaptivePerFeatureActivation::new(
                    in_features,
                    num_activations,
                ))
            }
            LayerKind::DualSlopeReLU => {
                let features = self.input_shape.streams[0];
                Box::new(crate::layers::DualSlopeReLU::new(features))
            }
            LayerKind::LearnableMish => {
                let features = self.input_shape.streams[0];
                Box::new(crate::layers::LearnableMish::new(features))
            }
            LayerKind::LearnableSoftplus => {
                let features = self.input_shape.streams[0];
                Box::new(crate::layers::LearnableSoftplus::new(features))
            }
            LayerKind::AdaptiveNormalization => {
                let features = self.input_shape.streams[0];
                Box::new(crate::layers::AdaptiveNormalization::new(features))
            }
            LayerKind::BatchRenorm1d => {
                let features = self.input_shape.streams[0];
                Box::new(crate::layers::BatchRenorm1d::new(features))
            }
            LayerKind::RMSNormWithLearnableEpsilon => {
                let features = self.input_shape.streams[0];
                Box::new(crate::layers::RMSNormWithLearnableEpsilon::new(features))
            }
            LayerKind::ConcreteDropout => {
                let temp = self.extra.get(0).copied().unwrap_or(0.1);
                let seed = self.extra.get(1).copied().unwrap_or(0.0) as u64;
                Box::new(crate::layers::ConcreteDropout::new_with_seed(temp, seed))
            }
            LayerKind::AdaptiveDropout => {
                let features = self.input_shape.streams[0];
                let seed = self.extra.get(0).copied().unwrap_or(0.0) as u64;
                Box::new(crate::layers::AdaptiveDropout::new_with_seed(features, seed))
            }
            LayerKind::LinearAttention => {
                let seq_len = self.extra.get(0).copied().unwrap_or(1.0) as usize;
                let d_model = self.extra.get(1).copied().unwrap_or(1.0) as usize;
                let min_heads = self.extra.get(2).copied().unwrap_or(1.0).max(1.0) as usize;
                let max_heads_raw = self.extra.get(3).copied().unwrap_or(min_heads as f32) as usize;
                let max_heads = max_heads_raw.max(min_heads);
                Box::new(crate::layers::LinearAttention::new(
                    seq_len,
                    d_model,
                    min_heads,
                    max_heads,
                ))
            }
            LayerKind::RelativePositionAttention => {
                let seq_len = self.extra.get(0).copied().unwrap_or(1.0) as usize;
                let d_model = self.extra.get(1).copied().unwrap_or(1.0) as usize;
                Box::new(crate::layers::RelativePositionAttention::new(seq_len, d_model))
            }
            LayerKind::IndRNN => {
                let input_dim = self.extra.get(0).copied().unwrap_or(1.0) as usize;
                let seq_len = self.extra.get(1).copied().unwrap_or(1.0) as usize;
                Box::new(crate::layers::IndRNN::new(input_dim, seq_len))
            }
            LayerKind::Mamba => {
                let seq_len = self.extra.get(0).copied().unwrap_or(1.0) as usize;
                let input_dim = self.extra.get(1).copied().unwrap_or(1.0) as usize;
                let state_dim = self.extra.get(2).copied().unwrap_or(1.0) as usize;
                Box::new(crate::layers::Mamba::new(seq_len, input_dim, state_dim))
            }
            LayerKind::FeatureFusion => {
                let in_features = self.input_shape.streams[0];
                let out_features = self.output_shape.streams[0];
                Box::new(crate::layers::FeatureFusion::new(in_features, out_features))
            }
            LayerKind::SpectrallyNormalizedLinear => {
                let in_features = self.input_shape.streams[0];
                let out_features = self.output_shape.streams[0];
                Box::new(crate::layers::SpectrallyNormalizedLinear::new(in_features, out_features))
            }
            LayerKind::SparseFeatureSelectionGate => {
                let features = self.input_shape.streams[0];
                Box::new(crate::layers::SparseFeatureSelectionGate::new(features))
            }
            LayerKind::MultiResolutionKANLinear => {
                let in_features = self.input_shape.streams[0];
                let out_features = self.output_shape.streams[0];
                Box::new(crate::layers::MultiResolutionKANLinear::new(in_features, out_features))
            }
            LayerKind::PerFeatureAttention => {
                let seq_len = self.extra.get(0).copied().unwrap_or(1.0) as usize;
                let d_model = self.extra.get(1).copied().unwrap_or(1.0) as usize;
                Box::new(crate::layers::PerFeatureAttention::new(seq_len, d_model))
            }
            _ => panic!("Unsupported layer kind for UniversalLayer: {:?}", self.kind),
        }
    }
}