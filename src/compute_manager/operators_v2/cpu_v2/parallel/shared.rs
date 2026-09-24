// src/compute_manager/cpu/parallel/shared.rs
//
// Общие структуры данных, разделяемые между forward- и backward-оркестраторами.

use std::sync::{Arc, Mutex};

use crate::compute_manager::core::dynamic_context::ChunkedContexts;
use crate::compute_manager::operators_v2::memory_v2::buffer::{MatrixBufferHandle, TempMatrixPool};
use crate::layers::UniversalLayer;
use crate::model_plan::param_store::ParamSlice;

/// Общие данные для всех forward-задач одного вызова.
pub(super) struct ForwardTaskShared {
    pub input: MatrixBufferHandle,
    pub output: MatrixBufferHandle,
    pub params: MatrixBufferHandle,
    pub layers: Arc<Vec<Box<dyn UniversalLayer>>>,
    pub slices: Arc<Vec<ParamSlice>>,
    pub pool: Arc<Mutex<TempMatrixPool>>,
    /// Длины реальных данных каждого примера **всего** батча. `None` —
    /// вход dense. `Some(lens)` — ragged; для каждого чанка берётся
    /// срез `lens[in_start..in_end]` из `chunk_ops`-плана.
    pub sample_lens: Option<Vec<usize>>,
}

/// Общие данные для всех backward-задач одного вызова.
pub(super) struct BackwardTaskShared {
    pub grad_output: MatrixBufferHandle,
    pub grad_input: MatrixBufferHandle,
    pub params: MatrixBufferHandle,
    pub grad_params: MatrixBufferHandle,
    pub layers: Arc<Vec<Box<dyn UniversalLayer>>>,
    pub slices: Arc<Vec<ParamSlice>>,
    pub contexts: ChunkedContexts,
    pub pool: Arc<Mutex<TempMatrixPool>>,
}