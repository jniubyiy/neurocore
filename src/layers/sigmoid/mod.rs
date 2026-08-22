// src/layers/sigmoid/mod.rs

pub mod sigmoid;

mod cpu;
mod gpu;

pub use sigmoid::Sigmoid;