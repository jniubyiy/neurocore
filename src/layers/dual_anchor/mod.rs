// src/layers/dual_anchor/mod.rs

pub mod dual_anchor;

pub mod gpu;   // <-- делаем модуль gpu публичным

mod cpu;

pub use dual_anchor::DualAnchor;