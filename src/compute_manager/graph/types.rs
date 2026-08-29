// src/compute_manager/graph/types.rs

use std::sync::Arc;

use crate::layers::buffered_context::BufferedContext;
use crate::model_plan::param_store::ParamSlice;
use crate::layers::UniversalLayer;

/// Контекст, сохраняемый слоями для обратного прохода.
#[derive(Clone)]
pub enum DynamicContext {
    /// Буферизованный контекст на основе `MatrixBufferHandle`.
    Buffered(BufferedContext),
}

/// Типы моделей вычислительного графа.
///
/// Модель представляет собой логическую группу слоёв или операцию,
/// которая может быть размещена на вычислительном устройстве независимо.
/// Ранее называлась "сегментом".
#[derive(Clone)]
pub enum Model {
    UniversalProcessor(
        Arc<Vec<Box<dyn UniversalLayer>>>,
        Vec<ParamSlice>,
        Option<Vec<usize>>,
    ),
    Unsqueeze(Vec<usize>),
    ReduceMean(Vec<usize>),
    SplitterConnector {
        dim_a: usize,
        dim_b: usize,
    },
    CombinerConnector {
        input_dims: Vec<usize>,
        #[allow(dead_code)]
        output_dim: usize,
    },
    Splitter {
        input_dim: usize,
        output_dims: Vec<usize>,
        slice: ParamSlice,
    },
    Combiner {
        input_dim: usize,
        output_dim: usize,
        slice: ParamSlice,
    },
}

