// src/dispatchers/mod.rs

pub mod common;
pub mod auto_model;

pub use auto_model::{
    MixedModel, DynamicTensor, DynamicContext,
    Dim1Processor, Dim2Processor, Dim3Processor, Dim4Processor, Dim5Processor,
};