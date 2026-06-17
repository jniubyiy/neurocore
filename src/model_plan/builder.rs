use crate::layers::Layer;
use crate::layers::layers1d::linear1d::LinearLayer;
use crate::layers::layers1d::relu1d::ReLULayer;
use crate::layers::layers1d::sigmoid1d::SigmoidLayer;
use crate::layers::layers1d::softmax1d::SoftmaxLayer;
use crate::model_plan::param_store::ParamSlice;

pub trait LayerBuilder {
    fn build(&self, slice: ParamSlice) -> Box<dyn Layer>;
    fn input_dim(&self) -> usize;
    fn output_dim(&self) -> usize;
    fn param_len(&self) -> usize;
}

pub struct LinearLayerBuilder {
    in_features: usize,
    out_features: usize,
}

impl LinearLayerBuilder {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        Self { in_features, out_features }
    }
}

impl LayerBuilder for LinearLayerBuilder {
    fn build(&self, _slice: ParamSlice) -> Box<dyn Layer> {
        Box::new(LinearLayer::new(self.in_features, self.out_features))
    }
    fn input_dim(&self) -> usize { self.in_features }
    fn output_dim(&self) -> usize { self.out_features }
    fn param_len(&self) -> usize { self.in_features * self.out_features + self.out_features }
}

pub struct ReLULayerBuilder;

impl LayerBuilder for ReLULayerBuilder {
    fn build(&self, _slice: ParamSlice) -> Box<dyn Layer> {
        Box::new(ReLULayer::new())
    }
    fn input_dim(&self) -> usize { 0 }
    fn output_dim(&self) -> usize { 0 }
    fn param_len(&self) -> usize { 0 }
}

pub struct SigmoidLayerBuilder;

impl LayerBuilder for SigmoidLayerBuilder {
    fn build(&self, _slice: ParamSlice) -> Box<dyn Layer> {
        Box::new(SigmoidLayer::new())
    }
    fn input_dim(&self) -> usize { 0 }
    fn output_dim(&self) -> usize { 0 }
    fn param_len(&self) -> usize { 0 }
}

pub struct SoftmaxLayerBuilder;

impl LayerBuilder for SoftmaxLayerBuilder {
    fn build(&self, _slice: ParamSlice) -> Box<dyn Layer> {
        Box::new(SoftmaxLayer::new())
    }
    fn input_dim(&self) -> usize { 0 }
    fn output_dim(&self) -> usize { 0 }
    fn param_len(&self) -> usize { 0 }
}