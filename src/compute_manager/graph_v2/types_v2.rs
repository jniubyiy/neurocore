// src/compute_manager/graph_v2/types_v2.rs

use std::sync::Arc;

use crate::compute_manager::jobs_v2::{
    ConnectorOpKind, DimOpKind, ForwardContextsV2,
};
use crate::compute_manager::operators_v2::memory_v2::buffer::MatrixBufferHandle;
use crate::layers::UniversalLayer;
use crate::model_plan::param_store::ParamSlice;

pub enum SegmentKindV2 {
    Universal {
        layers: Arc<Vec<Box<dyn UniversalLayer>>>,
        slices: Vec<ParamSlice>,
    },
    DimOp { kind: DimOpKind },
    Connector { kind: ConnectorOpKind },
}

pub struct SegmentV2 {
    pub index: usize,
    pub kind: SegmentKindV2,
    pub input_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
    pub stream_count: usize,
    pub stream_indices: Option<Vec<usize>>,
}

#[derive(Clone)]
pub enum SegmentForwardStateV2 {
    None,
    Universal { contexts: ForwardContextsV2 },
    Connector {
        inputs: Vec<MatrixBufferHandle>,
        pre: Vec<MatrixBufferHandle>,
    },
}

/// Кэш forward-прохода.
///
/// `sample_lens` — длины реальных данных каждого примера входного
/// батча. При ragged-входе `graph.adapter_pass` и другие потребители
/// могут узнать по этому полю, что вход был ragged, хотя сам слой
/// `AdaptiveSpaceCompress` читает длины из `BufferedContext`.
#[derive(Clone)]
pub struct ForwardCacheV2 {
    pub segment_states: Vec<SegmentForwardStateV2>,
    pub output: MatrixBufferHandle,
    pub batch: usize,
    pub sample_lens: Option<Vec<usize>>,
}

impl ForwardCacheV2 {
    #[inline]
    pub fn len(&self) -> usize {
        self.segment_states.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.segment_states.is_empty()
    }
}