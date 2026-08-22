// src/layers/softmax/mod.rs

pub mod softmax;

mod cpu;
mod gpu;

pub use softmax::Softmax;