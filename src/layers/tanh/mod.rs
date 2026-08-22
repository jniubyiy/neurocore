// src/layers/tanh/mod.rs

pub mod tanh;

mod cpu;
mod gpu;

pub use tanh::Tanh;