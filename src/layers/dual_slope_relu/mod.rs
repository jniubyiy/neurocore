// src/layers/dual_slope_relu/mod.rs

pub mod dual_slope_relu;

pub mod gpu;   // делаем модуль gpu публичным

mod cpu;

pub use dual_slope_relu::DualSlopeReLU;