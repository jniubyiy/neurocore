// ============================================================
// Файл: src/model_plan/blueprint.rs
// ============================================================

use crate::model_plan::dim::Dim;
use crate::model_plan::param_store::{ParamSlice, ParamStore};
use crate::layers::{
    Layer as Layer1D,
    LinearLayer, ReLULayer, SigmoidLayer, SoftmaxLayer, TanhLayer, MemoryLayer,
    SplitterConnector1D, CombinerConnector1D,
};
use crate::layers::{
    Layer2D,
    Linear2D, ReLU2D, Sigmoid2D, Softmax2D, Tanh2D,
};
use crate::layers::{
    Layer3D,
    Linear3D, ReLU3D, Sigmoid3D, Softmax3D, Tanh3D,
};
use crate::layers::{
    Layer4D,
    Linear4D, ReLU4D, Sigmoid4D, Softmax4D, Tanh4D,
};
use crate::layers::{
    Layer5D,
    Linear5D, ReLU5D, Sigmoid5D, Softmax5D, Tanh5D,
};

#[derive(Debug, Clone, PartialEq)]
pub enum LayerKind {
    Linear,
    ReLU,
    Sigmoid,
    Softmax,
    Memory,
    Tanh,
    Combiner,
    Splitter,
    Unsqueeze(usize),
    ReduceMean(usize),
    SplitterConnector,
    CombinerConnector,
}

#[derive(Debug, Clone)]
pub struct LayerBlueprint {
    pub input_dim: Dim,
    pub output_dim: Dim,
    pub kind: LayerKind,
    pub in_features: Vec<usize>,
    pub out_features: Vec<usize>,
    pub name: String,
}

impl LayerBlueprint {
    // -----------------------------------------------------------------------
    // Вычисление количества параметров слоя
    // -----------------------------------------------------------------------
    pub fn param_len(&self) -> usize {
        match &self.kind {
            LayerKind::Linear => {
                let in_dim = self.in_features[0];
                let out_dim = self.out_features[0];
                in_dim * out_dim + out_dim
            }
            LayerKind::Combiner => {
                let n = self.in_features[0];
                let m = self.out_features[0];
                2 * m * n + m
            }
            LayerKind::Splitter => {
                let n = self.in_features[0];
                let p = self.out_features[0];
                let q = self.out_features[1];
                p * n + q * n + p + q
            }
            LayerKind::Memory => {
                let in_dim = self.in_features[0];
                let out_dim = self.out_features[0];
                out_dim * (2 * in_dim + 1)
            }
            LayerKind::SplitterConnector | LayerKind::CombinerConnector => 0,
            LayerKind::Unsqueeze(_) | LayerKind::ReduceMean(_) => 0,
            _ => 0,
        }
    }

    // -----------------------------------------------------------------------
    // Конструкторы (сохраняем обратную совместимость)
    // -----------------------------------------------------------------------

    pub fn linear(dim: Dim, in_features: usize, out_features: usize) -> Self {
        Self {
            input_dim: dim,
            output_dim: dim,
            kind: LayerKind::Linear,
            in_features: vec![in_features],
            out_features: vec![out_features],
            name: String::new(),
        }
    }

    pub fn relu(dim: Dim, size: usize) -> Self {
        Self {
            input_dim: dim,
            output_dim: dim,
            kind: LayerKind::ReLU,
            in_features: vec![size],
            out_features: vec![size],
            name: String::new(),
        }
    }

    pub fn sigmoid(dim: Dim, size: usize) -> Self {
        Self {
            input_dim: dim,
            output_dim: dim,
            kind: LayerKind::Sigmoid,
            in_features: vec![size],
            out_features: vec![size],
            name: String::new(),
        }
    }

    pub fn softmax(dim: Dim, size: usize) -> Self {
        Self {
            input_dim: dim,
            output_dim: dim,
            kind: LayerKind::Softmax,
            in_features: vec![size],
            out_features: vec![size],
            name: String::new(),
        }
    }

    pub fn tanh(dim: Dim, size: usize) -> Self {
        Self {
            input_dim: dim,
            output_dim: dim,
            kind: LayerKind::Tanh,
            in_features: vec![size],
            out_features: vec![size],
            name: String::new(),
        }
    }

    pub fn unsqueeze(input_dim: Dim, axis: usize) -> Self {
        let output_dim = input_dim.next().expect("Unsqueeze: cannot increase dim above 5D");
        Self {
            input_dim,
            output_dim,
            kind: LayerKind::Unsqueeze(axis),
            in_features: Vec::new(),
            out_features: Vec::new(),
            name: String::new(),
        }
    }

    pub fn reduce_mean(input_dim: Dim, axis: usize) -> Self {
        let output_dim = input_dim.prev().expect("ReduceMean: cannot decrease dim below 1D");
        Self {
            input_dim,
            output_dim,
            kind: LayerKind::ReduceMean(axis),
            in_features: Vec::new(),
            out_features: Vec::new(),
            name: String::new(),
        }
    }

    // -----------------------------------------------------------------------
    // Создание слоёв без выделения памяти (для MixedModel)
    // -----------------------------------------------------------------------

    pub fn create_layer_1d(&self) -> Box<dyn Layer1D + Send + Sync> {
        assert_eq!(self.input_dim, Dim::Dim1);
        assert_eq!(self.output_dim, Dim::Dim1);
        match self.kind {
            LayerKind::Linear => Box::new(LinearLayer::new(self.in_features[0], self.out_features[0])),
            LayerKind::ReLU => Box::new(ReLULayer::new()),
            LayerKind::Sigmoid => Box::new(SigmoidLayer::new()),
            LayerKind::Softmax => Box::new(SoftmaxLayer::new()),
            LayerKind::Tanh => Box::new(TanhLayer::new()),
            LayerKind::Memory => Box::new(MemoryLayer::new(self.in_features[0], self.out_features[0])),
            LayerKind::SplitterConnector => Box::new(SplitterConnector1D::new(self.in_features[0], self.out_features.clone())),
            LayerKind::CombinerConnector => Box::new(CombinerConnector1D::new(self.in_features.clone())),
            _ => panic!("Unsupported layer kind for 1D"),
        }
    }

    pub fn create_layer_2d(&self) -> Box<dyn Layer2D + Send + Sync> {
        assert_eq!(self.input_dim, Dim::Dim2);
        assert_eq!(self.output_dim, Dim::Dim2);
        match self.kind {
            LayerKind::Linear => Box::new(Linear2D::new(self.in_features[0], self.out_features[0])),
            LayerKind::ReLU => Box::new(ReLU2D::new()),
            LayerKind::Sigmoid => Box::new(Sigmoid2D::new()),
            LayerKind::Softmax => Box::new(Softmax2D::new()),
            LayerKind::Tanh => Box::new(Tanh2D::new()),
            _ => panic!("Unsupported layer kind for 2D"),
        }
    }

    pub fn create_layer_3d(&self) -> Box<dyn Layer3D + Send + Sync> {
        assert_eq!(self.input_dim, Dim::Dim3);
        assert_eq!(self.output_dim, Dim::Dim3);
        match self.kind {
            LayerKind::Linear => Box::new(Linear3D::new(self.in_features[0], self.out_features[0])),
            LayerKind::ReLU => Box::new(ReLU3D::new()),
            LayerKind::Sigmoid => Box::new(Sigmoid3D::new()),
            LayerKind::Softmax => Box::new(Softmax3D::new()),
            LayerKind::Tanh => Box::new(Tanh3D::new()),
            _ => panic!("Unsupported layer kind for 3D"),
        }
    }

    pub fn create_layer_4d(&self) -> Box<dyn Layer4D + Send + Sync> {
        assert_eq!(self.input_dim, Dim::Dim4);
        assert_eq!(self.output_dim, Dim::Dim4);
        match self.kind {
            LayerKind::Linear => Box::new(Linear4D::new(self.in_features[0], self.out_features[0])),
            LayerKind::ReLU => Box::new(ReLU4D::new()),
            LayerKind::Sigmoid => Box::new(Sigmoid4D::new()),
            LayerKind::Softmax => Box::new(Softmax4D::new()),
            LayerKind::Tanh => Box::new(Tanh4D::new()),
            _ => panic!("Unsupported layer kind for 4D"),
        }
    }

    pub fn create_layer_5d(&self) -> Box<dyn Layer5D + Send + Sync> {
        assert_eq!(self.input_dim, Dim::Dim5);
        assert_eq!(self.output_dim, Dim::Dim5);
        match self.kind {
            LayerKind::Linear => Box::new(Linear5D::new(self.in_features[0], self.out_features[0])),
            LayerKind::ReLU => Box::new(ReLU5D::new()),
            LayerKind::Sigmoid => Box::new(Sigmoid5D::new()),
            LayerKind::Softmax => Box::new(Softmax5D::new()),
            LayerKind::Tanh => Box::new(Tanh5D::new()),
            _ => panic!("Unsupported layer kind for 5D"),
        }
    }

    // -----------------------------------------------------------------------
    // Старые build_layer_*d (совместимость) – используют create + allocate
    // -----------------------------------------------------------------------

    pub fn build_layer_1d(&self, store: &mut ParamStore) -> (Box<dyn Layer1D + Send + Sync>, ParamSlice) {
        let layer = self.create_layer_1d();
        let slice = store.allocate(self.param_len());
        (layer, slice)
    }

    pub fn build_layer_2d(&self, store: &mut ParamStore) -> (Box<dyn Layer2D + Send + Sync>, ParamSlice) {
        let layer = self.create_layer_2d();
        let slice = store.allocate(self.param_len());
        (layer, slice)
    }

    pub fn build_layer_3d(&self, store: &mut ParamStore) -> (Box<dyn Layer3D + Send + Sync>, ParamSlice) {
        let layer = self.create_layer_3d();
        let slice = store.allocate(self.param_len());
        (layer, slice)
    }

    pub fn build_layer_4d(&self, store: &mut ParamStore) -> (Box<dyn Layer4D + Send + Sync>, ParamSlice) {
        let layer = self.create_layer_4d();
        let slice = store.allocate(self.param_len());
        (layer, slice)
    }

    pub fn build_layer_5d(&self, store: &mut ParamStore) -> (Box<dyn Layer5D + Send + Sync>, ParamSlice) {
        let layer = self.create_layer_5d();
        let slice = store.allocate(self.param_len());
        (layer, slice)
    }
}