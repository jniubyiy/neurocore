// src/compute_manager/graph/types.rs

use std::sync::Arc;

use crate::layers::mat_context::MatContext;
use crate::model_plan::param_store::ParamSlice;
use crate::tensor::{Tensor2D, Tensor3D, Tensor4D, Tensor5D};

use crate::layers::UniversalLayer;

#[derive(Clone)]
pub enum DynamicContext {
    Mat(MatContext),
}

pub enum DynamicBatchTensor {
    Dim1(Vec<Tensor2D>),
    Dim2(Vec<Tensor3D>),
    Dim3(Vec<Tensor4D>),
    Dim4(Vec<Tensor5D>),
}

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

