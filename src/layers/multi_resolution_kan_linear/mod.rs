// src/layers/multi_resolution_kan_linear/mod.rs

pub mod multi_resolution_kan_linear;

pub mod gpu;   // публичный, но реализация вызывает panic

mod cpu;

pub use multi_resolution_kan_linear::MultiResolutionKANLinear;