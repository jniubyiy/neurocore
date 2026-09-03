// src/layers/learnable_softplus/mod.rs

pub mod learnable_softplus;

pub mod gpu;   // делаем модуль gpu публичным

mod cpu;

pub use learnable_softplus::LearnableSoftplus;