// src/layers/sigmoid/mod.rs

pub mod sigmoid;

pub mod gpu;   // <-- делаем модуль gpu публичным

mod cpu;

pub use sigmoid::Sigmoid;