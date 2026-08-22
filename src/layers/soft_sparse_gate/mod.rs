// src/layers/soft_sparse_gate/mod.rs

pub mod soft_sparse_gate;

mod cpu;
mod gpu;

pub use soft_sparse_gate::SoftSparseGate;