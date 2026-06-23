// src/model_plan/blueprint/create_layers.rs

use crate::layers::UniversalLayer;
use crate::layers::{Linear, ReLU, Sigmoid, Softmax, Tanh, Memory};
use super::layer_kind::LayerKind;
use super::blueprint_struct::LayerBlueprint;

impl LayerBlueprint {
    pub fn create_universal_layer(&self) -> Box<dyn UniversalLayer> {
        match self.kind {
            LayerKind::Linear => Box::new(Linear::new(self.in_features[0], self.out_features[0])),
            LayerKind::ReLU => Box::new(ReLU::new()),
            LayerKind::Sigmoid => Box::new(Sigmoid::new()),
            LayerKind::Softmax => Box::new(Softmax::new()),
            LayerKind::Tanh => Box::new(Tanh::new()),
            LayerKind::Memory => Box::new(Memory::new(self.in_features[0], self.out_features[0])),
            _ => panic!("Unsupported layer kind for UniversalLayer"),
        }
    }
}