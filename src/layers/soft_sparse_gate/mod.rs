// src/layers/soft_sparse_gate/mod.rs

pub mod soft_sparse_gate;

pub mod gpu;   // <-- делаем модуль gpu публичным

mod cpu;

pub use soft_sparse_gate::SoftSparseGate;