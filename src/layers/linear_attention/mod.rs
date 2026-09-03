// src/layers/linear_attention/mod.rs

pub mod linear_attention;

pub mod gpu;   // публичный, но реализация вызывает panic

mod cpu;

pub use linear_attention::LinearAttention;