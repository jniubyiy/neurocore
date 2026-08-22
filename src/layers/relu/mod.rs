// src/layers/relu/mod.rs

pub mod relu;

mod cpu;
mod gpu;

pub use relu::ReLU;