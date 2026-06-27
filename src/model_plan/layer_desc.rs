// src/model_plan/layer_desc.rs

use super::dim::Dim;
use super::blueprint::LayerKind;

pub trait IntoSizes {
    fn into_vec(self) -> Vec<usize>;
}

impl IntoSizes for &[usize] {
    fn into_vec(self) -> Vec<usize> { self.to_vec() }
}

impl<const N: usize> IntoSizes for &[usize; N] {
    fn into_vec(self) -> Vec<usize> { self.to_vec() }
}

// Обобщённая реализация для кортежа из двух слайсов/массивов
impl<A: AsRef<[usize]>, B: AsRef<[usize]>> IntoSizes for (A, B) {
    fn into_vec(self) -> Vec<usize> {
        let mut v = Vec::new();
        v.extend_from_slice(self.0.as_ref());
        v.extend_from_slice(self.1.as_ref());
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
    pub extra: Vec<f32>,   // дополнительные гиперпараметры (alpha, temperature и т.п.)
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
            extra: Vec::new(),
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

    pub fn extra(mut self, values: Vec<f32>) -> Self {
        self.extra = values;
        self
    }

    // Вспомогательные конструкторы для распространённых слоёв

    pub fn linear(dim: Dim, in_features: usize, out_features: usize) -> Self {
        Self {
            input_dim: dim,
            output_dim: dim,
            kind: LayerKind::Linear,
            in_features: vec![in_features],
            out_features: vec![out_features],
            name: String::new(),
            extra: Vec::new(),
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
            extra: Vec::new(),
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
            extra: Vec::new(),
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
            extra: Vec::new(),
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
            extra: Vec::new(),
        }
    }

    pub fn leaky_relu(dim: Dim, size: usize, alpha: f32) -> Self {
        Self {
            input_dim: dim,
            output_dim: dim,
            kind: LayerKind::LeakyReLU,
            in_features: vec![size],
            out_features: vec![size],
            name: String::new(),
            extra: vec![alpha],
        }
    }

    pub fn identity(dim: Dim, size: usize) -> Self {
        Self {
            input_dim: dim,
            output_dim: dim,
            kind: LayerKind::Identity,
            in_features: vec![size],
            out_features: vec![size],
            name: String::new(),
            extra: Vec::new(),
        }
    }

    pub fn soft_sparse_gate(dim: Dim, size: usize, temperature: f32) -> Self {
        Self {
            input_dim: dim,
            output_dim: dim,
            kind: LayerKind::SoftSparseGate,
            in_features: vec![size],
            out_features: vec![size],
            name: String::new(),
            extra: vec![temperature],
        }
    }

    pub fn soft_keep_gate(dim: Dim, size: usize, temperature: f32) -> Self {
        Self {
            input_dim: dim,
            output_dim: dim,
            kind: LayerKind::SoftKeepGate,
            in_features: vec![size],
            out_features: vec![size],
            name: String::new(),
            extra: vec![temperature],
        }
    }

    pub fn dual_anchor(dim: Dim, size: usize) -> Self {
        Self {
            input_dim: dim,
            output_dim: dim,
            kind: LayerKind::DualAnchor,
            in_features: vec![size],
            out_features: vec![size],
            name: String::new(),
            extra: Vec::new(),
        }
    }

    pub fn unsqueeze(input_dim: Dim, target_dims: Vec<usize>) -> Self {
        let output_dim = match input_dim {
            Dim::Dim1 => Dim::Dim2,
            Dim::Dim2 => Dim::Dim3,
            Dim::Dim3 => Dim::Dim4,
            Dim::Dim4 => panic!("Cannot unsqueeze from 4D"),
        };
        Self {
            input_dim,
            output_dim,
            kind: LayerKind::Unsqueeze(target_dims.clone()),
            in_features: vec![target_dims.iter().product()],
            out_features: target_dims,
            name: String::new(),
            extra: Vec::new(),
        }
    }

    pub fn reduce_mean(input_dim: Dim, target_dims: Vec<usize>) -> Self {
        let output_dim = match input_dim {
            Dim::Dim2 => Dim::Dim1,
            Dim::Dim3 => Dim::Dim2,
            Dim::Dim4 => Dim::Dim3,
            _ => panic!("ReduceMean requires at least 2D input"),
        };
        Self {
            input_dim,
            output_dim,
            kind: LayerKind::ReduceMean(target_dims.clone()),
            in_features: target_dims.clone(),
            out_features: vec![target_dims.iter().product()],
            name: String::new(),
            extra: Vec::new(),
        }
    }

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
            LayerKind::SoftSparseGate | LayerKind::SoftKeepGate => {
                // обучаемые пороги по числу признаков
                self.in_features[0]
            }
            LayerKind::DualAnchor => {
                // min + max + alpha
                2 * self.in_features[0] + 1
            }
            LayerKind::LeakyReLU | LayerKind::Identity => 0,
            LayerKind::SplitterConnector | LayerKind::CombinerConnector => 0,
            LayerKind::Unsqueeze(_) | LayerKind::ReduceMean(_) => 0,
            _ => 0,
        }
    }

    pub fn create_universal_layer(&self) -> Box<dyn crate::layers::UniversalLayer> {
        match self.kind {
            LayerKind::Linear => Box::new(crate::layers::Linear::new(self.in_features[0], self.out_features[0])),
            LayerKind::ReLU => Box::new(crate::layers::ReLU::new()),
            LayerKind::Sigmoid => Box::new(crate::layers::Sigmoid::new()),
            LayerKind::Softmax => Box::new(crate::layers::Softmax::new()),
            LayerKind::Tanh => Box::new(crate::layers::Tanh::new()),
            LayerKind::Memory => Box::new(crate::layers::Memory::new(self.in_features[0], self.out_features[0])),
            LayerKind::LeakyReLU => {
                let alpha = self.extra.get(0).copied().unwrap_or(0.01);
                Box::new(crate::layers::LeakyReLU::new(alpha))
            }
            LayerKind::Identity => Box::new(crate::layers::Identity::new()),
            LayerKind::SoftSparseGate => {
                let temp = self.extra.get(0).copied().unwrap_or(1.0);
                Box::new(crate::layers::SoftSparseGate::new(self.in_features[0], temp))
            }
            LayerKind::SoftKeepGate => {
                let temp = self.extra.get(0).copied().unwrap_or(1.0);
                Box::new(crate::layers::SoftKeepGate::new(self.in_features[0], temp))
            }
            LayerKind::DualAnchor => {
                Box::new(crate::layers::DualAnchor::new(self.in_features[0], self.out_features[0]))
            }
            _ => panic!("Unsupported layer kind for UniversalLayer"),
        }
    }
}

// Макросы
#[macro_export]
macro_rules! create_models {
    // Вариант с явным указанием устройства:
    // create_models!(func1, func2, ..., device = some_device)
    // Обратите внимание: Device должен быть импортирован в область вызова,
    // например: use neurocore::compute_manager::Device;
    ( $( $func:path ),+ , device = $device:expr $(,)? ) => {
        ( $(
            $crate::model_plan::Plan::from_layer_descs($func())
                .expect("Invalid model description")
                .build_with_device($device.clone())
        ,)+ )
    };
    // Вариант без устройства (по умолчанию CPU)
    ( $( $func:path ),+ $(,)? ) => {
        ( $(
            $crate::model_plan::Plan::from_layer_descs($func())
                .expect("Invalid model description")
                .build()
        ,)+ )
    };
}

#[macro_export]
macro_rules! create_losses {
    ( $( $func:path ),+ $(,)? ) => {
        ( $(
            $func().build()
        ,)+ )
    };
}

#[macro_export]
macro_rules! create_optimizers {
    ( $( ($model:expr, $desc_func:path) ),+ $(,)? ) => {
        ( $(
            $model.create_optimizer($desc_func().build_chain())
        ,)+ )
    };
}