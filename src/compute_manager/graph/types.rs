// src/compute_manager/graph/types.rs

use std::sync::Arc;

use crate::layers::context1d::LayerContext1D;
use crate::layers::context2d::LayerContext;
use crate::layers::context3d::LayerContext3D;
use crate::layers::context4d::LayerContext4D;
use crate::model_plan::param_store::ParamSlice;
use crate::tensor::{Tensor2D, Tensor3D, Tensor4D, Tensor5D};

use crate::layers::UniversalLayer;

/// Динамический контекст слоя (объединяет все размерности).
#[derive(Clone)]
pub enum DynamicContext {
    Ctx1D(LayerContext1D),
    Ctx2D(LayerContext),
    Ctx3D(LayerContext3D),
    Ctx4D(LayerContext4D),
}

/// Батч тензоров с динамической размерностью.
pub enum DynamicBatchTensor {
    Dim1(Vec<Tensor2D>),
    Dim2(Vec<Tensor3D>),
    Dim3(Vec<Tensor4D>),
    Dim4(Vec<Tensor5D>),
}

pub(crate) enum Segment {
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