// src/compute_manager/gpu/compute/mod.rs

pub mod base;
pub mod matmul;
pub mod activation;
pub mod softmax;
pub mod linear;
pub mod dim_ops;
pub mod loss_cubes;
pub mod optimizer;
pub mod custom_layers;
pub mod splitter_combiner;

pub use base::GpuCompute;