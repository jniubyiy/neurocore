// src/layers/linear/mod.rs

pub mod linear;

pub mod gpu;   // <-- делаем модуль gpu публичным

mod cpu;

pub use linear::Linear;