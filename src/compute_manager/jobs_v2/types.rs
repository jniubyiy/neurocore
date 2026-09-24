// src/compute_manager/jobs_v2/types.rs

use std::sync::Arc;

use crate::compute_manager::core::dynamic_context::ChunkedContexts;
use crate::compute_manager::operators_v2::gpu_v2::GpuCompute;
use crate::compute_manager::operators_v2::memory_v2::buffer::MatrixBufferHandle;
use crate::compute_manager::operators_v2::memory_v2::types::MemoryDeviceKind;
use crate::compute_manager::DynamicContext;

use crate::layers::UniversalLayer;
use crate::loss_plan::LossExpr;
use crate::model_plan::param_store::ParamSlice;
use crate::optimizer_plan::OptimizerDesc;
use crate::training_plan::Initializer;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum JobKind {
    Migrate,
    ForwardSegment,
    BackwardSegment,
    Loss,
    OptimizerModifyGrads,
    OptimizerApplyUpdate,
    DimOp,
    ConnectorOp,
    ParamInit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OperatorKind {
    Memory,
    Cpu,
    Gpu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct JobHandle {
    pub id: u64,
    pub kind: JobKind,
    pub operator: OperatorKind,
}

#[derive(Clone)]
pub enum ForwardContextsV2 {
    Sequential(Vec<DynamicContext>),
    Chunked {
        contexts: ChunkedContexts,
        layout: Vec<(usize, usize, usize)>,
    },
}

impl ForwardContextsV2 {
    #[inline]
    pub fn num_chunks(&self) -> usize {
        match self {
            ForwardContextsV2::Sequential(_) => 1,
            ForwardContextsV2::Chunked { contexts, .. } => contexts.len(),
        }
    }

    #[inline]
    pub fn first_chunk(&self) -> Vec<DynamicContext> {
        match self {
            ForwardContextsV2::Sequential(v) => v.clone(),
            ForwardContextsV2::Chunked { contexts, .. } => {
                contexts.first().cloned().unwrap_or_default()
            }
        }
    }
}

/// Прямой проход одного UniversalProcessor-сегмента.
pub struct ForwardSegmentJob {
    pub segment_index: usize,
    pub layers: Arc<Vec<Box<dyn UniversalLayer>>>,
    pub slices: Vec<ParamSlice>,
    pub params: MatrixBufferHandle,
    pub input: MatrixBufferHandle,
    /// Длины реальных данных каждого примера входного чанка.
    ///
    /// * `None` — вход плотный: все строки имеют длину `input.cols()`.
    /// * `Some(lens)` — вход ragged: `lens[r]` — реальная длина примера `r`.
    ///   Слои, поддерживающие ragged (в частности `AdaptiveSpaceCompress`),
    ///   читают это поле и обрабатывают только реальную часть.
    pub sample_lens: Option<Vec<usize>>,
}

pub struct BackwardSegmentJob {
    pub segment_index: usize,
    pub layers: Arc<Vec<Box<dyn UniversalLayer>>>,
    pub slices: Vec<ParamSlice>,
    pub params: MatrixBufferHandle,
    pub grad_params: MatrixBufferHandle,
    pub grad_output: MatrixBufferHandle,
    pub contexts: ForwardContextsV2,
}

pub struct LossJob {
    pub expr: Arc<LossExpr>,
    pub pred: MatrixBufferHandle,
    pub target: MatrixBufferHandle,
}

pub struct OptimizerModifyGradsJob {
    pub buffer_idx: usize,
    pub params: MatrixBufferHandle,
    pub grads: MatrixBufferHandle,
    pub optimizer: OptimizerDesc,
    pub gpu_compute: Option<Arc<GpuCompute>>,
}

pub struct OptimizerApplyUpdateJob {
    pub buffer_idx: usize,
    pub params: MatrixBufferHandle,
    pub grads: MatrixBufferHandle,
    pub optimizer: OptimizerDesc,
    pub gpu_compute: Option<Arc<GpuCompute>>,
}

pub struct MigrateJob {
    pub handle: MatrixBufferHandle,
    pub target: MemoryDeviceKind,
}

#[derive(Debug, Clone)]
pub enum DimOpKind {
    Unsqueeze(Vec<usize>),
    ReduceMean(Vec<usize>),
}

pub struct DimOpJob {
    pub kind: DimOpKind,
    pub input: MatrixBufferHandle,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectorDirection {
    Forward,
    Backward,
}

#[derive(Debug, Clone)]
pub enum ConnectorOpKind {
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

pub struct ConnectorOpJob {
    pub direction: ConnectorDirection,
    pub kind: ConnectorOpKind,
    pub inputs: Vec<MatrixBufferHandle>,
    pub params: Option<MatrixBufferHandle>,
    pub grad_params: Option<MatrixBufferHandle>,
    pub saved: Vec<MatrixBufferHandle>,
}

pub struct ParamInitJob {
    pub buffer_idx: usize,
    pub params: MatrixBufferHandle,
    pub initializer: Initializer,
    pub seed: Option<u64>,
    pub precomputed: Option<Vec<f32>>,
}

pub enum Job {
    Migrate(MigrateJob),
    ForwardSegment(ForwardSegmentJob),
    BackwardSegment(BackwardSegmentJob),
    Loss(LossJob),
    OptimizerModifyGrads(OptimizerModifyGradsJob),
    OptimizerApplyUpdate(OptimizerApplyUpdateJob),
    DimOp(DimOpJob),
    ConnectorOp(ConnectorOpJob),
    ParamInit(ParamInitJob),
}

impl Job {
    #[inline]
    pub fn kind(&self) -> JobKind {
        match self {
            Job::Migrate(_) => JobKind::Migrate,
            Job::ForwardSegment(_) => JobKind::ForwardSegment,
            Job::BackwardSegment(_) => JobKind::BackwardSegment,
            Job::Loss(_) => JobKind::Loss,
            Job::OptimizerModifyGrads(_) => JobKind::OptimizerModifyGrads,
            Job::OptimizerApplyUpdate(_) => JobKind::OptimizerApplyUpdate,
            Job::DimOp(_) => JobKind::DimOp,
            Job::ConnectorOp(_) => JobKind::ConnectorOp,
            Job::ParamInit(_) => JobKind::ParamInit,
        }
    }
}

pub enum JobResult {
    Unit,
    Forward {
        output: MatrixBufferHandle,
        contexts: ForwardContextsV2,
    },
    Backward {
        grad_input: MatrixBufferHandle,
    },
    Loss {
        value: f32,
        grad_pred: MatrixBufferHandle,
    },
    Buffer(MatrixBufferHandle),
    Buffers(Vec<MatrixBufferHandle>),
    Failed(String),
}

impl JobResult {
    #[inline]
    pub fn is_failed(&self) -> bool {
        matches!(self, JobResult::Failed(_))
    }

    #[inline]
    pub fn into_failed_message(self) -> Option<String> {
        match self {
            JobResult::Failed(msg) => Some(msg),
            _ => None,
        }
    }
}