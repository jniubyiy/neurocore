// src/compute_manager/cpu/parallel/shared.rs
//
// Общие структуры данных и отладочные переключатели, разделяемые
// между forward- и backward-оркестраторами.

use std::sync::{Arc, Mutex};

use once_cell::sync::Lazy;

use crate::compute_manager::graph::types::ChunkedContexts;
use crate::compute_manager::matrix_buffer::{MatrixBufferHandle, TempMatrixPool};
use crate::layers::UniversalLayer;
use crate::model_plan::param_store::ParamSlice;

/// Отладочный переключатель параллельного прохода.
///
/// Включается переменной окружения `NEUROCORE_DEBUG_PARALLEL=1`.
/// При включении в stderr печатается подробный трейс диспетчеризации
/// forward/backward по чанкам и слоям.
pub(super) static PARALLEL_DEBUG: Lazy<bool> =
    Lazy::new(|| std::env::var("NEUROCORE_DEBUG_PARALLEL").is_ok());

/// Общие данные для всех forward-задач одного вызова.
pub(super) struct ForwardTaskShared {
    pub input: MatrixBufferHandle,
    pub output: MatrixBufferHandle,
    pub params: MatrixBufferHandle,
    pub layers: Arc<Vec<Box<dyn UniversalLayer>>>,
    pub slices: Arc<Vec<ParamSlice>>,
    pub pool: Arc<Mutex<TempMatrixPool>>,
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