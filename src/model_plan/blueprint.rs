use crate::model_plan::dim::Dim;
use crate::model_plan::param_store::{ParamSlice, ParamStore};

use crate::layers::{
    LinearLayer, ReLULayer, SigmoidLayer, SoftmaxLayer, TanhLayer, MemoryLayer,
    Layer as Layer1D,
};
use crate::layers::{
    Linear2D, ReLU2D, Sigmoid2D, Softmax2D, Tanh2D,
    Layer2D,
};
use crate::layers::{
    Linear3D, ReLU3D, Sigmoid3D, Softmax3D, Tanh3D,
    Layer3D,
};
use crate::layers::{
    Linear4D, ReLU4D, Sigmoid4D, Softmax4D, Tanh4D,
    Layer4D,
};
use crate::layers::{
    Linear5D, ReLU5D, Sigmoid5D, Softmax5D, Tanh5D,
    Layer5D,
};

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

    pub fn build_layer_1d(&self, store: &mut ParamStore) -> (Box<dyn Layer1D + Send + Sync>, ParamSlice) {
        match self.kind {
            LayerKind::Linear => {
                let len = self.param_len();
                let slice = store.allocate(len);
                (Box::new(LinearLayer::new(self.in_features, self.out_features)), slice)
            }
            LayerKind::ReLU => (Box::new(ReLULayer::new()), ParamSlice::new(0, 0)),
            LayerKind::Sigmoid => (Box::new(SigmoidLayer::new()), ParamSlice::new(0, 0)),
            LayerKind::Softmax => (Box::new(SoftmaxLayer::new()), ParamSlice::new(0, 0)),
            LayerKind::Tanh => (Box::new(TanhLayer::new()), ParamSlice::new(0, 0)),
            LayerKind::Memory => {
                let len = self.param_len();
                let slice = store.allocate(len);
                (Box::new(MemoryLayer::new(self.in_features, self.out_features)), slice)
            }
        }
    }

    pub fn build_layer_2d(&self, store: &mut ParamStore) -> (Box<dyn Layer2D + Send + Sync>, ParamSlice) {
        match self.kind {
            LayerKind::Linear => {
                let len = self.param_len();
                let slice = store.allocate(len);
                (Box::new(Linear2D::new(self.in_features, self.out_features)), slice)
            }
            LayerKind::ReLU => (Box::new(ReLU2D::new(self.in_features)), ParamSlice::new(0, 0)),
            LayerKind::Sigmoid => (Box::new(Sigmoid2D::new(self.in_features)), ParamSlice::new(0, 0)),
            LayerKind::Softmax => (Box::new(Softmax2D::new(self.in_features)), ParamSlice::new(0, 0)),
            LayerKind::Tanh => (Box::new(Tanh2D::new(self.in_features)), ParamSlice::new(0, 0)),
            LayerKind::Memory => panic!("Memory2D временно отключён"),
        }
    }

    pub fn build_layer_3d(&self, store: &mut ParamStore) -> (Box<dyn Layer3D + Send + Sync>, ParamSlice) {
        match self.kind {
            LayerKind::Linear => {
                let len = self.param_len();
                let slice = store.allocate(len);
                (Box::new(Linear3D::new(self.in_features, self.out_features)), slice)
            }
            LayerKind::ReLU => (Box::new(ReLU3D::new(self.in_features)), ParamSlice::new(0, 0)),
            LayerKind::Sigmoid => (Box::new(Sigmoid3D::new(self.in_features)), ParamSlice::new(0, 0)),
            LayerKind::Softmax => (Box::new(Softmax3D::new(self.in_features)), ParamSlice::new(0, 0)),
            LayerKind::Tanh => (Box::new(Tanh3D::new(self.in_features)), ParamSlice::new(0, 0)),
            LayerKind::Memory => panic!("Memory3D временно отключён"),
        }
    }

    pub fn build_layer_4d(&self, store: &mut ParamStore) -> (Box<dyn Layer4D + Send + Sync>, ParamSlice) {
        match self.kind {
            LayerKind::Linear => {
                let len = self.param_len();
                let slice = store.allocate(len);
                (Box::new(Linear4D::new(self.in_features, self.out_features)), slice)
            }
            LayerKind::ReLU => (Box::new(ReLU4D::new(self.in_features)), ParamSlice::new(0, 0)),
            LayerKind::Sigmoid => (Box::new(Sigmoid4D::new(self.in_features)), ParamSlice::new(0, 0)),
            LayerKind::Softmax => (Box::new(Softmax4D::new(self.in_features)), ParamSlice::new(0, 0)),
            LayerKind::Tanh => (Box::new(Tanh4D::new(self.in_features)), ParamSlice::new(0, 0)),
            LayerKind::Memory => panic!("Memory4D временно отключён"),
        }
    }

    pub fn build_layer_5d(&self, store: &mut ParamStore) -> (Box<dyn Layer5D + Send + Sync>, ParamSlice) {
        match self.kind {
            LayerKind::Linear => {
                let len = self.param_len();
                let slice = store.allocate(len);
                (Box::new(Linear5D::new(self.in_features, self.out_features)), slice)
            }
            LayerKind::ReLU => (Box::new(ReLU5D::new(self.in_features)), ParamSlice::new(0, 0)),
            LayerKind::Sigmoid => (Box::new(Sigmoid5D::new(self.in_features)), ParamSlice::new(0, 0)),
            LayerKind::Softmax => (Box::new(Softmax5D::new(self.in_features)), ParamSlice::new(0, 0)),
            LayerKind::Tanh => (Box::new(Tanh5D::new(self.in_features)), ParamSlice::new(0, 0)),
            LayerKind::Memory => panic!("Memory5D временно отключён"),
        }
    }
}