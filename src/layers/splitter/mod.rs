// src/layers/splitter/mod.rs

pub mod splitter;

pub mod gpu;   // <-- делаем модуль gpu публичным

mod cpu;

pub use splitter::Splitter;