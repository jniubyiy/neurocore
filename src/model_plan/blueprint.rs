use std::sync::Arc;
use crate::model_plan::dim::Dim;
use crate::model_plan::param_store::{ParamSlice, ParamStore};

use crate::layers::{
    LinearLayer, ReLULayer, SigmoidLayer, SoftmaxLayer, TanhLayer, MemoryLayer,
    Layer as Layer1D,
};
use crate::layers::{
    Linear2D, ReLU2D, Sigmoid2D, Softmax2D, Tanh2D, Memory2D,
    Layer2D,
};

/// Проверяет, является ли число степенью двойки (1,2,4,8,…).
/// Если нет – паникует с понятным сообщением.
pub fn assert_power_of_two(x: usize) {
    assert!(
        x > 0 && (x & (x - 1)) == 0,
        "Размер должен быть степенью двойки (1,2,4,8,…), получено {}",
        x
    );
}

pub fn is_power_of_two(x: usize) -> bool {
    x > 0 && (x & (x - 1)) == 0
}

#[derive(Debug, Clone, PartialEq)]
pub enum LayerKind {
    Linear,
    ReLU,
    Sigmoid,
    Softmax,
    Memory,
    Tanh,
}

pub struct LayerBlueprint {
    pub dim: Dim,
    pub kind: LayerKind,
    pub in_features: usize,
    pub out_features: usize,
}

impl LayerBlueprint {
    pub fn linear(dim: Dim, in_features: usize, out_features: usize) -> Self {
        assert_power_of_two(in_features);
        assert_power_of_two(out_features);
        Self { dim, kind: LayerKind::Linear, in_features, out_features }
    }
    pub fn relu(dim: Dim, size: usize) -> Self {
        assert_power_of_two(size);
        Self { dim, kind: LayerKind::ReLU, in_features: size, out_features: size }
    }
    pub fn sigmoid(dim: Dim, size: usize) -> Self {
        assert_power_of_two(size);
        Self { dim, kind: LayerKind::Sigmoid, in_features: size, out_features: size }
    }
    pub fn softmax(dim: Dim, size: usize) -> Self {
        assert_power_of_two(size);
        Self { dim, kind: LayerKind::Softmax, in_features: size, out_features: size }
    }
    pub fn tanh(dim: Dim, size: usize) -> Self {
        assert_power_of_two(size);
        Self { dim, kind: LayerKind::Tanh, in_features: size, out_features: size }
    }
    pub fn memory(dim: Dim, in_features: usize, out_features: usize) -> Self {
        assert_power_of_two(in_features);
        assert_power_of_two(out_features);
        Self { dim, kind: LayerKind::Memory, in_features, out_features }
    }

    pub fn param_len(&self) -> usize {
        match self.kind {
            LayerKind::Linear => self.in_features * self.out_features + self.out_features,
            LayerKind::Memory => self.out_features * (2 * self.in_features + 1),
            _ => 0,
        }
    }

    pub fn build_layer_1d(&self, store: &mut ParamStore) -> (Arc<dyn Layer1D + Send + Sync>, ParamSlice) {
        match self.kind {
            LayerKind::Linear => {
                let len = self.param_len();
                let slice = store.allocate(len);
                (Arc::new(LinearLayer::new(self.in_features, self.out_features, slice)), slice)
            }
            LayerKind::ReLU => (Arc::new(ReLULayer::new()), ParamSlice::new(0, 0)),
            LayerKind::Sigmoid => (Arc::new(SigmoidLayer::new()), ParamSlice::new(0, 0)),
            LayerKind::Softmax => (Arc::new(SoftmaxLayer::new()), ParamSlice::new(0, 0)),
            LayerKind::Tanh => (Arc::new(TanhLayer::new()), ParamSlice::new(0, 0)),
            LayerKind::Memory => {
                let len = self.param_len();
                let slice = store.allocate(len);
                (Arc::new(MemoryLayer::new(self.in_features, self.out_features, slice)), slice)
            }
        }
    }

    pub fn build_layer_2d(&self, store: &mut ParamStore) -> (Arc<dyn Layer2D + Send + Sync>, ParamSlice) {
        match self.kind {
            LayerKind::Linear => {
                let len = self.param_len();
                let slice = store.allocate(len);
                (Arc::new(Linear2D::new(self.in_features, self.out_features, slice)), slice)
            }
            LayerKind::ReLU => (Arc::new(ReLU2D::new(self.in_features)), ParamSlice::new(0, 0)),
            LayerKind::Sigmoid => (Arc::new(Sigmoid2D::new(self.in_features)), ParamSlice::new(0, 0)),
            LayerKind::Softmax => (Arc::new(Softmax2D::new(self.in_features)), ParamSlice::new(0, 0)),
            LayerKind::Tanh => (Arc::new(Tanh2D::new(self.in_features)), ParamSlice::new(0, 0)),
            LayerKind::Memory => {
                let len = self.param_len();
                let slice = store.allocate(len);
                (Arc::new(Memory2D::new(self.in_features, self.out_features, slice)), slice)
            }
        }
    }
}