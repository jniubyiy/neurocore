// ============================================================
// Файл: src/model_plan/layer_desc.rs
// ============================================================

use super::dim::Dim;
use super::blueprint::{LayerBlueprint, LayerKind};

pub trait IntoSizes {
    fn into_vec(self) -> Vec<usize>;
}

impl IntoSizes for &[usize] {
    fn into_vec(self) -> Vec<usize> { self.to_vec() }
}

impl<const N: usize> IntoSizes for &[usize; N] {
    fn into_vec(self) -> Vec<usize> { self.to_vec() }
}

impl IntoSizes for (&[usize], &[usize]) {
    fn into_vec(self) -> Vec<usize> {
        let mut v = Vec::new();
        v.extend_from_slice(self.0);
        v.extend_from_slice(self.1);
        v
    }
}

#[derive(Debug, Clone)]
pub struct LayerDesc {
    pub name: String,
    pub kind: LayerKind,
    pub input_dim: Dim,
    pub output_dim: Dim,
    pub in_features: Vec<usize>,
    pub out_features: Vec<usize>,
}

impl LayerDesc {
    pub fn new(name: &str, kind: LayerKind, dim: Dim) -> Self {
        Self {
            name: name.to_string(),
            kind,
            input_dim: dim,
            output_dim: dim,
            in_features: Vec::new(),
            out_features: Vec::new(),
        }
    }

    pub fn unsqueeze(name: &str, input_dim: Dim, axis: usize) -> Self {
        let output_dim = input_dim.next().expect("Unsqueeze: нельзя повысить размерность выше 5D");
        Self {
            name: name.to_string(),
            kind: LayerKind::Unsqueeze(axis),
            input_dim,
            output_dim,
            in_features: Vec::new(),
            out_features: Vec::new(),
        }
    }

    pub fn reduce_mean(name: &str, input_dim: Dim, axis: usize) -> Self {
        let output_dim = input_dim.prev().expect("ReduceMean: нельзя понизить размерность ниже 1D");
        Self {
            name: name.to_string(),
            kind: LayerKind::ReduceMean(axis),
            input_dim,
            output_dim,
            in_features: Vec::new(),
            out_features: Vec::new(),
        }
    }

    pub fn splitter_connector(name: &str, dim: Dim, input_dim: usize, output_dims: Vec<usize>) -> Self {
        Self {
            name: name.to_string(),
            kind: LayerKind::SplitterConnector,
            input_dim: dim,
            output_dim: dim,
            in_features: vec![input_dim],
            out_features: output_dims,
        }
    }

    pub fn combiner_connector(name: &str, dim: Dim, input_dims: Vec<usize>) -> Self {
        let output_dim = input_dims.iter().sum();
        Self {
            name: name.to_string(),
            kind: LayerKind::CombinerConnector,
            input_dim: dim,
            output_dim: dim,
            in_features: input_dims,
            out_features: vec![output_dim],
        }
    }

    pub fn input<I: IntoSizes>(mut self, dim: Dim, sizes: I) -> Self {
        self.input_dim = dim;
        self.in_features = sizes.into_vec();
        self
    }

    pub fn output<O: IntoSizes>(mut self, dim: Dim, sizes: O) -> Self {
        self.output_dim = dim;
        self.out_features = sizes.into_vec();
        self
    }

    pub fn into_blueprint(self) -> LayerBlueprint {
        LayerBlueprint {
            input_dim: self.input_dim,
            output_dim: self.output_dim,
            kind: self.kind,
            in_features: self.in_features,
            out_features: self.out_features,
            name: self.name,
        }
    }
}