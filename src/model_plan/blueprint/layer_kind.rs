// src/model_plan/blueprint/layer_kind.rs

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
    Unsqueeze(Vec<usize>),
    ReduceMean(Vec<usize>),
    SplitterConnector,
    CombinerConnector,
}