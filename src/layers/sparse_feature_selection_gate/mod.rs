// src/layers/sparse_feature_selection_gate/mod.rs

pub mod sparse_feature_selection_gate;

pub mod gpu;   // делаем модуль gpu публичным

mod cpu;

pub use sparse_feature_selection_gate::SparseFeatureSelectionGate;