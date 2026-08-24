// src/layers/combiner/mod.rs

pub mod combiner;

pub mod gpu;   // <-- делаем модуль gpu публичным

mod cpu;

pub use combiner::Combiner;