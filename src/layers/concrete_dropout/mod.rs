// src/layers/concrete_dropout/mod.rs

pub mod concrete_dropout;
pub mod gpu;
mod cpu;

pub use concrete_dropout::ConcreteDropout;