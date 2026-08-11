// src/compute_manager/graph/types.rs

use std::sync::Arc;

use crate::layers::mat_context::MatContext;
use crate::model_plan::param_store::ParamSlice;
use crate::layers::UniversalLayer;

/// Контекст, сохраняемый слоями для обратного прохода.
#[derive(Clone)]
pub enum DynamicContext {
    Mat(MatContext),
}

/// Типы сегментов вычислительного графа.
#[derive(Clone)]
pub enum Segment {
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

