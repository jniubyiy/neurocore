// src/layers/mat_context.rs

use faer::Mat;

/// Информация о слое, используемая планировщиком и для отладки.
#[derive(Debug, Clone)]
pub struct LayerInfo {
    pub layer_type: String,
    pub input_dim1s: Vec<usize>,
    pub output_dim1s: Vec<usize>,
    pub param_count: usize,
    pub param_start_index: Option<usize>,
}

/// Матричный контекст, сохраняемый слоем во время прямого прохода
/// для последующего использования в обратном проходе.
///
/// Все поля — матрицы `faer::Mat<f32>`, никаких тензоров.
#[derive(Clone)]
pub enum MatContext {
    Linear {
        input: Mat<f32>,
    },
    ReLU {
        input: Mat<f32>,
    },
    Sigmoid {
        output: Mat<f32>,
    },
    Tanh {
        output: Mat<f32>,
    },
    Softmax {
        output: Mat<f32>,
    },
    Memory {
        input: Mat<f32>,
    },
    Combiner {
        input_a: Mat<f32>,
        input_b: Mat<f32>,
        pre_act: Mat<f32>,
    },
    Splitter {
        input: Mat<f32>,
        pre_a: Mat<f32>,
        pre_b: Mat<f32>,
    },
    SplitterConnector {
        input: Mat<f32>,
    },
    CombinerConnector {
        inputs: Vec<Mat<f32>>,
    },
    LeakyReLU {
        input: Mat<f32>,
    },
    SoftSparseGate {
        input: Mat<f32>,
    },
    SoftKeepGate {
        input: Mat<f32>,
    },
    DualAnchor1D {
        input: Mat<f32>,
    },
    Identity {
        input: Mat<f32>,
    },
    Unsqueeze {
        input: Mat<f32>,
    },
    ReduceMean {
        input: Mat<f32>,
    },
}