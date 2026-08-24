// src/layers/soft_keep_gate/mod.rs

pub mod soft_keep_gate;

pub mod gpu;   // <-- делаем модуль gpu публичным

mod cpu;

pub use soft_keep_gate::SoftKeepGate;