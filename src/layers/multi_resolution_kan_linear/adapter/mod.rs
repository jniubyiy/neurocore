// src/layers/multi_resolution_kan_linear/adapter/mod.rs

//! Градиентный адаптер слоя `MultiResolutionKANLinear`
//! (MIGRATION_PLAN.md §7).
//!
//! Полное описание математики, режимов и результатов экспериментов —
//! в документации модуля [`adapter`].

mod adapter;

pub use adapter::MultiResolutionKANLinearAdapter;