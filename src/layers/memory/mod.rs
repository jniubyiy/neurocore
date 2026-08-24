// src/layers/memory/mod.rs

pub mod memory;

pub mod gpu;   // <-- делаем модуль gpu публичным

mod cpu;

pub use memory::Memory;