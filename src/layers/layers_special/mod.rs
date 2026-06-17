pub mod reduce_dim;
pub mod expand_dim;

pub use reduce_dim::ReduceMean;
pub use expand_dim::Unsqueeze;

/// Трейт для уменьшения размерности (без якобианов, без параметров).
pub trait DimReduce<InputTensor, OutputTensor> {
    fn reduce(&self, input: &InputTensor) -> OutputTensor;
}

/// Трейт для увеличения размерности (без якобианов, без параметров).
pub trait DimExpand<InputTensor, OutputTensor> {
    fn expand(&self, input: &InputTensor) -> OutputTensor;
}




