// src/plans/model_plan/param_store/mod.rs

//! Модуль хранения параметров модели, организованных по сегментам.
//!
//! В отличие от предыдущей версии с единым буфером `BufferedParamStore`,
//! новая архитектура использует отдельные [`ParamBuffer`] для каждого
//! сегмента модели. Это позволяет независимо управлять размещением
//! параметров на разных устройствах (CPU, GPU, SSD) и выполнять
//! миграцию между ними без остановки вычислений.

mod param_buffer;
mod param_store;
mod slice;

pub use param_buffer::ParamBuffer;
pub use param_store::ParamStore;
pub use slice::ParamSlice;