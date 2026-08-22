// src/layers/sigmoid/mod.rs

pub mod sigmoid;

pub use sigmoid::Sigmoid;

mod cpu;
mod gpu;