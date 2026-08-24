// src/layers/tanh/mod.rs

pub mod tanh;

pub mod gpu;   // <-- делаем модуль gpu публичным

mod cpu;

pub use tanh::Tanh;