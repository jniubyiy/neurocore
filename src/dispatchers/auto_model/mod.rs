// src/dispatchers/auto_model/mod.rs

pub mod dim1d;
pub mod dim2d;
pub mod dim3d;
pub mod dim4d;
pub mod dim5d;
pub mod dim_change;
pub mod mixed;

pub use dim1d::Dim1Processor;
pub use dim2d::Dim2Processor;
pub use dim3d::Dim3Processor;
pub use dim4d::Dim4Processor;
pub use dim5d::Dim5Processor;
pub use dim_change::DynamicTensor;
pub use mixed::{MixedModel, DynamicContext};