// src/layers/leaky_relu/mod.rs

pub mod leaky_relu;

pub mod gpu;   // <-- делаем модуль gpu публичным

mod cpu;

pub use leaky_relu::LeakyReLU;