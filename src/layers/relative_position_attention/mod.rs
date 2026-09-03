// src/layers/relative_position_attention/mod.rs

pub mod relative_position_attention;

pub mod gpu;   // публичный, но реализация вызывает panic

mod cpu;

pub use relative_position_attention::RelativePositionAttention;