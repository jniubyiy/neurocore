// src/layers/batch_renorm/mod.rs

pub mod batch_renorm;

pub mod gpu;   // делаем модуль gpu публичным

mod cpu;

pub use batch_renorm::BatchRenorm1d;