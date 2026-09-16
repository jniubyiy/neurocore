// src/compute_manager/gpu/processor/layers/mod.rs

//! Пер-категорийные обработчики слоёв.
//!
//! Каждый модуль экспортирует две функции:
//! - `forward(gpu, layer, input, params_handle, slice) -> Option<(out, ctx)>`
//! - `backward(gpu, layer, ctx, grad_output, params_handle, slice, grad_params_handle) -> Option<grad_input>`
//!
//! Функция возвращает `None`, если переданный слой не относится к её категории.
//! Оркестраторы (`super::forward`, `super::backward`) последовательно опрашивают
//! все модули, пока один из них не вернёт `Some`.
//!
//! Единственное исключение — `memory::forward`: ему дополнительно передаётся
//! `&mut memory_idx`, поскольку состояния Memory на GPU индексируются по
//! порядковому номеру слоя в сегменте.

pub mod linear;
pub mod activation;
pub mod memory;
pub mod gate;
pub mod anchor;
pub mod norm;
pub mod dropout;
pub mod kan;
pub mod feature_fusion;
pub mod spectral_norm;
pub mod attention;
pub mod recurrent;