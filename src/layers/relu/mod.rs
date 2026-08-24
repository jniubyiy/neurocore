// src/layers/relu/mod.rs

pub mod relu;

pub mod gpu;   // <-- делаем модуль gpu публичным

mod cpu;

pub use relu::ReLU;