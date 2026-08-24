// src/layers/softmax/mod.rs

pub mod softmax;

pub mod gpu;   // <-- делаем модуль gpu публичным

mod cpu;

pub use softmax::Softmax;