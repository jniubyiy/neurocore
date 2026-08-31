// src/compute_manager/graph/types.rs

use std::sync::Arc;

use crate::layers::buffered_context::BufferedContext;
use crate::layers::UniversalLayer;
use crate::model_plan::param_store::ParamSlice;

/// Контекст, сохраняемый слоями для обратного прохода.
#[derive(Clone)]
pub enum DynamicContext {
    /// Буферизованный контекст на основе `MatrixBufferHandle`.
    Buffered(BufferedContext),
}

/// Контексты обратного прохода, сгруппированные по чанкам.
/// Внешний вектор содержит по одному элементу на каждый чанк батча.
/// Каждый элемент — вектор контекстов слоёв для соответствующего чанка.
pub type ChunkedContexts = Vec<Vec<DynamicContext>>;

/// Типы моделей вычислительного графа.
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

