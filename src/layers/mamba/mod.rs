// src/layers/mamba/mod.rs

pub mod mamba;

pub mod gpu;   // публичный, но реализация вызывает panic

mod cpu;

pub use mamba::Mamba;