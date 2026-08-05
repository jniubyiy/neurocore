// src/plans/model_plan/blueprint/layer_kind.rs

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
    Unsqueeze,
    ReduceMean,
    SplitterConnector,
    CombinerConnector,
    LeakyReLU,
    Identity,
    SoftSparseGate,
    SoftKeepGate,
    DualAnchor,
}