// src/compute_manager/cpu/parallel.rs

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::compute_manager::cpu::WorkerPool;
use crate::compute_manager::executor::Executor;
use crate::compute_manager::graph::types::{ChunkedContexts, DynamicContext};
use crate::compute_manager::matrix_buffer::{MatrixBufferHandle, TempMatrixPool};
use crate::layers::buffered_context::BufferedContext;
use crate::layers::{
    UniversalLayer, UniversalLayerBuffered,
    Linear, ReLU, Sigmoid, Tanh, LeakyReLU, Identity, Softmax,
    Memory, SoftSparseGate, SoftKeepGate, DualAnchor, AdaptivePerFeatureActivation,
    DualSlopeReLU, LearnableMish, LearnableSoftplus, RMSNormWithLearnableEpsilon,
    AdaptiveDropout, FeatureFusion, SparseFeatureSelectionGate, MultiResolutionKANLinear,
    AdaptiveNormalization, BatchRenorm1d, ConcreteDropout, IndRNN, Mamba,
    SpectrallyNormalizedLinear,
};
use crate::model_plan::param_store::ParamSlice;

// ============================================================================
//  Отслеживание состояния чанков
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ChunkStatus {
    Pending,
    Assigned,
    InProgress,
    Done,
}

struct ChunkState {
    chunk_id: usize,
    start: usize,
    end: usize,
    status: ChunkStatus,
    assigned_worker: Option<usize>,
    physical_worker: Option<usize>,
    started_at: Option<Instant>,
    finished_at: Option<Instant>,
    duration_ns: u64,
}

/// Событийный трекер чанков.
///
/// Работает без таймеров и фоновых потоков: переходы состояний
/// (`Pending → Assigned → InProgress → Done`) фиксируются воркерами
/// непосредственно в моменты, когда эти переходы происходят.
///
/// Назначение:
/// * накопление статистики выполнения (длительности, распределение по воркерам);
/// * возможность по запросу сформировать дамп текущего состояния
///   (`dump()`), например, при разборе инцидентов или по запросу пользователя.
///
/// Трекер сам по себе не отслеживает «зависания» — понятие застоя требует
/// сравнения с ходом времени, а внутренние таймеры в ядре не используются.
#[allow(dead_code)]
struct ChunkTracker {
    chunks: Vec<ChunkState>,
    created_at: Instant,
    last_change: Instant,
    total_changes: usize,
}

impl ChunkTracker {
    fn new(chunks: &[(usize, usize, usize)]) -> Self {
        let now = Instant::now();
        let entries = chunks
            .iter()
            .enumerate()
            .map(|(chunk_id, &(start, _size, end))| ChunkState {
                chunk_id,
                start,
                end,
                status: ChunkStatus::Pending,
                assigned_worker: None,
                physical_worker: None,
                started_at: None,
                finished_at: None,
                duration_ns: 0,
            })
            .collect();
        Self {
            chunks: entries,
            created_at: now,
            last_change: now,
            total_changes: 0,
        }
    }

    fn mark_assigned(&mut self, chunk_id: usize, logical_worker_id: usize) {
        if let Some(c) = self.chunks.get_mut(chunk_id) {
            c.status = ChunkStatus::Assigned;
            c.assigned_worker = Some(logical_worker_id);
            self.last_change = Instant::now();
            self.total_changes += 1;
        }
    }

    fn mark_in_progress(&mut self, chunk_id: usize, physical_worker_id: usize) {
        if let Some(c) = self.chunks.get_mut(chunk_id) {
            c.status = ChunkStatus::InProgress;
            c.physical_worker = Some(physical_worker_id);
            c.started_at = Some(Instant::now());
            self.last_change = Instant::now();
            self.total_changes += 1;
        }
    }

    fn mark_done(&mut self, chunk_id: usize, duration_ns: u64) {
        if let Some(c) = self.chunks.get_mut(chunk_id) {
            c.status = ChunkStatus::Done;
            c.finished_at = Some(Instant::now());
            c.duration_ns = duration_ns;
            self.last_change = Instant::now();
            self.total_changes += 1;
        }
    }

    #[allow(dead_code)]
    fn total_changes(&self) -> usize {
        self.total_changes
    }

    /// Формирует текстовый дамп текущего состояния трекера.
    ///
    /// Метод чисто диагностический, не запускает никаких таймеров
    /// и не порождает потоков. Его может вызвать внешний код в любой
    /// момент, если ему нужно понять, что происходит с чанками.
    #[allow(dead_code)]
    fn dump(&self) -> String {
        let elapsed = self.created_at.elapsed();
        let mut s = String::new();

        let n_total = self.chunks.len();
        let n_done = self.chunks.iter().filter(|c| c.status == ChunkStatus::Done).count();
        let n_in_progress = self
            .chunks
            .iter()
            .filter(|c| c.status == ChunkStatus::InProgress)
            .count();
        let n_assigned = self
            .chunks
            .iter()
            .filter(|c| c.status == ChunkStatus::Assigned)
            .count();
        let n_pending = self
            .chunks
            .iter()
            .filter(|c| c.status == ChunkStatus::Pending)
            .count();

        let _ = writeln!(s, "=== ChunkTracker dump @ T+{:.3}s ===", elapsed.as_secs_f64());
        let _ = writeln!(
            s,
            "total={} | done={} | in_progress={} | assigned={} | pending={}",
            n_total, n_done, n_in_progress, n_assigned, n_pending
        );
        let _ = writeln!(
            s,
            "last_change: T+{:.3}s ({} changes total)",
            self.last_change.duration_since(self.created_at).as_secs_f64(),
            self.total_changes
        );

        let _ = writeln!(s, "--- done ---");
        let mut any = false;
        for c in self.chunks.iter().filter(|c| c.status == ChunkStatus::Done) {
            any = true;
            let _ = writeln!(
                s,
                "  chunk {:>3}  range [{:>3}..{:<3})  assigned={:?}  physical={:?}  dur={:.3}us",
                c.chunk_id,
                c.start,
                c.end,
                c.assigned_worker,
                c.physical_worker,
                c.duration_ns as f64 / 1000.0
            );
        }
        if !any {
            let _ = writeln!(s, "  (none)");
        }

        let _ = writeln!(s, "--- in_progress ---");
        any = false;
        for c in self.chunks.iter().filter(|c| c.status == ChunkStatus::InProgress) {
            any = true;
            let running = c
                .started_at
                .map(|a| format!("{:.3}s", a.elapsed().as_secs_f64()))
                .unwrap_or_else(|| "?".into());
            let _ = writeln!(
                s,
                "  chunk {:>3}  range [{:>3}..{:<3})  assigned={:?}  physical={:?}  running={}",
                c.chunk_id, c.start, c.end, c.assigned_worker, c.physical_worker, running
            );
        }
        if !any {
            let _ = writeln!(s, "  (none)");
        }

        let _ = writeln!(s, "--- assigned (not started) ---");
        any = false;
        for c in self.chunks.iter().filter(|c| c.status == ChunkStatus::Assigned) {
            any = true;
            let _ = writeln!(
                s,
                "  chunk {:>3}  range [{:>3}..{:<3})  assigned_worker={:?}",
                c.chunk_id, c.start, c.end, c.assigned_worker
            );
        }
        if !any {
            let _ = writeln!(s, "  (none)");
        }

        let _ = writeln!(s, "--- pending ---");
        any = false;
        for c in self.chunks.iter().filter(|c| c.status == ChunkStatus::Pending) {
            any = true;
            let _ = writeln!(s, "  chunk {:>3}  range [{:>3}..{:<3})", c.chunk_id, c.start, c.end);
        }
        if !any {
            let _ = writeln!(s, "  (none)");
        }

        let _ = writeln!(s, "--- logical worker stats ---");
        let mut lmap: BTreeMap<usize, (usize, usize, usize, Vec<usize>)> = BTreeMap::new();
        for c in &self.chunks {
            if let Some(w) = c.assigned_worker {
                let e = lmap.entry(w).or_insert((0, 0, 0, Vec::new()));
                match c.status {
                    ChunkStatus::Done => e.0 += 1,
                    ChunkStatus::InProgress => e.1 += 1,
                    ChunkStatus::Assigned => e.2 += 1,
                    ChunkStatus::Pending => {}
                }
                e.3.push(c.chunk_id);
            }
        }
        if lmap.is_empty() {
            let _ = writeln!(s, "  (none)");
        } else {
            for (w, (d, ip, a, ids)) in lmap {
                let _ = writeln!(
                    s,
                    "  logical worker {}: done={}, in_progress={}, assigned={}, chunks={:?}",
                    w, d, ip, a, ids
                );
            }
        }

        let _ = writeln!(s, "--- physical worker stats ---");
        let mut pmap: BTreeMap<usize, (usize, usize, Vec<usize>)> = BTreeMap::new();
        for c in &self.chunks {
            if let Some(w) = c.physical_worker {
                let e = pmap.entry(w).or_insert((0, 0, Vec::new()));
                match c.status {
                    ChunkStatus::Done => e.0 += 1,
                    ChunkStatus::InProgress => e.1 += 1,
                    _ => {}
                }
                e.2.push(c.chunk_id);
            }
        }
        if pmap.is_empty() {
            let _ = writeln!(s, "  (no physical worker has touched any chunk)");
        } else {
            for (w, (d, ip, ids)) in pmap {
                let _ = writeln!(
                    s,
                    "  physical worker {}: done={}, in_progress={}, chunks={:?}",
                    w, d, ip, ids
                );
            }
        }

        s
    }
}

// ============================================================================
//  Общие структуры задач
// ============================================================================

struct ForwardTaskShared {
    input: MatrixBufferHandle,
    output: MatrixBufferHandle,
    params: MatrixBufferHandle,
    layers: Arc<Vec<Box<dyn UniversalLayer>>>,
    slices: Arc<Vec<ParamSlice>>,
    pool: Arc<Mutex<TempMatrixPool>>,
}

struct BackwardTaskShared {
    grad_output: MatrixBufferHandle,
    grad_input: MatrixBufferHandle,
    params: MatrixBufferHandle,
    grad_params: MatrixBufferHandle,
    layers: Arc<Vec<Box<dyn UniversalLayer>>>,
    slices: Arc<Vec<ParamSlice>>,
    contexts: ChunkedContexts,
    pool: Arc<Mutex<TempMatrixPool>>,
}

// ============================================================================
//  Вспомогательные функции работы с чанками
// ============================================================================

pub(crate) fn extract_chunk(
    input: &MatrixBufferHandle,
    start: usize,
    end: usize,
    pool: &mut TempMatrixPool,
) -> MatrixBufferHandle {
    let rows_total = input.rows();
    let cols = input.cols();
    assert!(end <= rows_total && start < end);

    let chunk_rows = end - start;
    let chunk = pool.acquire(chunk_rows, cols);

    let src_guard = input.read();
    let src = src_guard.as_slice().expect("CPU buffer");
    let mut dst_guard = chunk.write();
    let dst = dst_guard.as_slice_mut().expect("CPU buffer");

    for c in 0..cols {
        for r in 0..chunk_rows {
            dst[c * chunk_rows + r] = src[c * rows_total + start + r];
        }
    }
    chunk
}

pub(crate) fn write_chunk(
    output: &MatrixBufferHandle,
    chunk: &MatrixBufferHandle,
    start: usize,
) {
    let out_rows = output.rows();
    let cols = output.cols();
    let chunk_rows = chunk.rows();
    assert_eq!(cols, chunk.cols());
    assert!(start + chunk_rows <= out_rows);

    let mut out_guard = output.write();
    let out_slice = out_guard.as_slice_mut().expect("CPU buffer");
    let chunk_guard = chunk.read();
    let chunk_slice = chunk_guard.as_slice().expect("CPU buffer");

    for c in 0..cols {
        for r in 0..chunk_rows {
            out_slice[c * out_rows + start + r] = chunk_slice[c * chunk_rows + r];
        }
    }
}

// ============================================================================
//  Размерности слоёв
// ============================================================================

fn get_output_features(layer: &Box<dyn UniversalLayer>, input: &MatrixBufferHandle) -> usize {
    if let Some(l) = layer.as_linear() {
        <Linear as UniversalLayerBuffered>::output_features(l)
    } else if let Some(l) = layer.as_feature_fusion() {
        <FeatureFusion as UniversalLayerBuffered>::output_features(l)
    } else if let Some(l) = layer.as_multi_resolution_kan_linear() {
        <MultiResolutionKANLinear as UniversalLayerBuffered>::output_features(l)
    } else if let Some(l) = layer.as_spectral_norm_linear() {
        <SpectrallyNormalizedLinear as UniversalLayerBuffered>::output_features(l)
    } else {
        input.cols()
    }
}

fn get_input_features(layer: &Box<dyn UniversalLayer>, grad_output: &MatrixBufferHandle) -> usize {
    if let Some(l) = layer.as_linear() {
        <Linear as UniversalLayerBuffered>::input_features(l)
    } else if let Some(l) = layer.as_feature_fusion() {
        <FeatureFusion as UniversalLayerBuffered>::input_features(l)
    } else if let Some(l) = layer.as_multi_resolution_kan_linear() {
        <MultiResolutionKANLinear as UniversalLayerBuffered>::input_features(l)
    } else if let Some(l) = layer.as_spectral_norm_linear() {
        <SpectrallyNormalizedLinear as UniversalLayerBuffered>::input_features(l)
    } else {
        grad_output.cols()
    }
}

// ============================================================================
//  Диспетчеризация слоёв
// ============================================================================

fn call_forward_buffered(
    layer: &Box<dyn UniversalLayer>,
    input: &MatrixBufferHandle,
    output: &MatrixBufferHandle,
    params: &MatrixBufferHandle,
    slice: &ParamSlice,
) {
    if let Some(l) = layer.as_linear() {
        <Linear as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice);
    } else if let Some(l) = layer.as_relu() {
        <ReLU as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice);
    } else if let Some(l) = layer.as_sigmoid() {
        <Sigmoid as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice);
    } else if let Some(l) = layer.as_tanh() {
        <Tanh as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice);
    } else if let Some(l) = layer.as_leaky_relu() {
        <LeakyReLU as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice);
    } else if let Some(l) = layer.as_identity() {
        <Identity as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice);
    } else if let Some(l) = layer.as_softmax() {
        <Softmax as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice);
    } else if let Some(l) = layer.as_memory() {
        <Memory as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice);
    } else if let Some(l) = layer.as_soft_sparse_gate() {
        <SoftSparseGate as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice);
    } else if let Some(l) = layer.as_soft_keep_gate() {
        <SoftKeepGate as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice);
    } else if let Some(l) = layer.as_dual_anchor() {
        <DualAnchor as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice);
    } else if let Some(l) = layer.as_adaptive_activation() {
        <AdaptivePerFeatureActivation as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice);
    } else if let Some(l) = layer.as_dual_slope_relu() {
        <DualSlopeReLU as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice);
    } else if let Some(l) = layer.as_learnable_mish() {
        <LearnableMish as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice);
    } else if let Some(l) = layer.as_learnable_softplus() {
        <LearnableSoftplus as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice);
    } else if let Some(l) = layer.as_rms_norm_learnable_eps() {
        <RMSNormWithLearnableEpsilon as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice);
    } else if let Some(l) = layer.as_adaptive_dropout() {
        <AdaptiveDropout as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice);
    } else if let Some(l) = layer.as_feature_fusion() {
        <FeatureFusion as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice);
    } else if let Some(l) = layer.as_sparse_feature_selection_gate() {
        <SparseFeatureSelectionGate as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice);
    } else if let Some(l) = layer.as_multi_resolution_kan_linear() {
        <MultiResolutionKANLinear as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice);
    } else if let Some(l) = layer.as_adaptive_normalization() {
        <AdaptiveNormalization as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice);
    } else if let Some(l) = layer.as_batch_renorm() {
        <BatchRenorm1d as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice);
    } else if let Some(l) = layer.as_concrete_dropout() {
        <ConcreteDropout as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice);
    } else if let Some(l) = layer.as_ind_rnn() {
        <IndRNN as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice);
    } else if let Some(l) = layer.as_mamba() {
        <Mamba as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice);
    } else if let Some(l) = layer.as_spectral_norm_linear() {
        <SpectrallyNormalizedLinear as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice);
    } else {
        unreachable!("Unsupported layer in parallel forward");
    }
}

fn call_backward_buffered(
    layer: &Box<dyn UniversalLayer>,
    ctx: &DynamicContext,
    grad_output: &MatrixBufferHandle,
    grad_input: &MatrixBufferHandle,
    params: &MatrixBufferHandle,
    slice: &ParamSlice,
    grad_params: &MatrixBufferHandle,
) {
    if let Some(l) = layer.as_linear() {
        <Linear as UniversalLayerBuffered>::backward_buffered(l, ctx, grad_output, grad_input, params, slice, grad_params);
    } else if let Some(l) = layer.as_relu() {
        <ReLU as UniversalLayerBuffered>::backward_buffered(l, ctx, grad_output, grad_input, params, slice, grad_params);
    } else if let Some(l) = layer.as_sigmoid() {
        <Sigmoid as UniversalLayerBuffered>::backward_buffered(l, ctx, grad_output, grad_input, params, slice, grad_params);
    } else if let Some(l) = layer.as_tanh() {
        <Tanh as UniversalLayerBuffered>::backward_buffered(l, ctx, grad_output, grad_input, params, slice, grad_params);
    } else if let Some(l) = layer.as_leaky_relu() {
        <LeakyReLU as UniversalLayerBuffered>::backward_buffered(l, ctx, grad_output, grad_input, params, slice, grad_params);
    } else if let Some(l) = layer.as_identity() {
        <Identity as UniversalLayerBuffered>::backward_buffered(l, ctx, grad_output, grad_input, params, slice, grad_params);
    } else if let Some(l) = layer.as_softmax() {
        <Softmax as UniversalLayerBuffered>::backward_buffered(l, ctx, grad_output, grad_input, params, slice, grad_params);
    } else if let Some(l) = layer.as_memory() {
        <Memory as UniversalLayerBuffered>::backward_buffered(l, ctx, grad_output, grad_input, params, slice, grad_params);
    } else if let Some(l) = layer.as_soft_sparse_gate() {
        <SoftSparseGate as UniversalLayerBuffered>::backward_buffered(l, ctx, grad_output, grad_input, params, slice, grad_params);
    } else if let Some(l) = layer.as_soft_keep_gate() {
        <SoftKeepGate as UniversalLayerBuffered>::backward_buffered(l, ctx, grad_output, grad_input, params, slice, grad_params);
    } else if let Some(l) = layer.as_dual_anchor() {
        <DualAnchor as UniversalLayerBuffered>::backward_buffered(l, ctx, grad_output, grad_input, params, slice, grad_params);
    } else if let Some(l) = layer.as_adaptive_activation() {
        <AdaptivePerFeatureActivation as UniversalLayerBuffered>::backward_buffered(l, ctx, grad_output, grad_input, params, slice, grad_params);
    } else if let Some(l) = layer.as_dual_slope_relu() {
        <DualSlopeReLU as UniversalLayerBuffered>::backward_buffered(l, ctx, grad_output, grad_input, params, slice, grad_params);
    } else if let Some(l) = layer.as_learnable_mish() {
        <LearnableMish as UniversalLayerBuffered>::backward_buffered(l, ctx, grad_output, grad_input, params, slice, grad_params);
    } else if let Some(l) = layer.as_learnable_softplus() {
        <LearnableSoftplus as UniversalLayerBuffered>::backward_buffered(l, ctx, grad_output, grad_input, params, slice, grad_params);
    } else if let Some(l) = layer.as_rms_norm_learnable_eps() {
        <RMSNormWithLearnableEpsilon as UniversalLayerBuffered>::backward_buffered(l, ctx, grad_output, grad_input, params, slice, grad_params);
    } else if let Some(l) = layer.as_adaptive_dropout() {
        <AdaptiveDropout as UniversalLayerBuffered>::backward_buffered(l, ctx, grad_output, grad_input, params, slice, grad_params);
    } else if let Some(l) = layer.as_feature_fusion() {
        <FeatureFusion as UniversalLayerBuffered>::backward_buffered(l, ctx, grad_output, grad_input, params, slice, grad_params);
    } else if let Some(l) = layer.as_sparse_feature_selection_gate() {
        <SparseFeatureSelectionGate as UniversalLayerBuffered>::backward_buffered(l, ctx, grad_output, grad_input, params, slice, grad_params);
    } else if let Some(l) = layer.as_multi_resolution_kan_linear() {
        <MultiResolutionKANLinear as UniversalLayerBuffered>::backward_buffered(l, ctx, grad_output, grad_input, params, slice, grad_params);
    } else if let Some(l) = layer.as_adaptive_normalization() {
        <AdaptiveNormalization as UniversalLayerBuffered>::backward_buffered(l, ctx, grad_output, grad_input, params, slice, grad_params);
    } else if let Some(l) = layer.as_batch_renorm() {
        <BatchRenorm1d as UniversalLayerBuffered>::backward_buffered(l, ctx, grad_output, grad_input, params, slice, grad_params);
    } else if let Some(l) = layer.as_concrete_dropout() {
        <ConcreteDropout as UniversalLayerBuffered>::backward_buffered(l, ctx, grad_output, grad_input, params, slice, grad_params);
    } else if let Some(l) = layer.as_ind_rnn() {
        <IndRNN as UniversalLayerBuffered>::backward_buffered(l, ctx, grad_output, grad_input, params, slice, grad_params);
    } else if let Some(l) = layer.as_mamba() {
        <Mamba as UniversalLayerBuffered>::backward_buffered(l, ctx, grad_output, grad_input, params, slice, grad_params);
    } else if let Some(l) = layer.as_spectral_norm_linear() {
        <SpectrallyNormalizedLinear as UniversalLayerBuffered>::backward_buffered(l, ctx, grad_output, grad_input, params, slice, grad_params);
    } else {
        unreachable!("Unsupported layer in parallel backward");
    }
}

fn build_buffered_context(
    layer: &Box<dyn UniversalLayer>,
    input: &MatrixBufferHandle,
    output: &MatrixBufferHandle,
    pool_guard: &mut TempMatrixPool,
) -> BufferedContext {
    if layer.as_linear().is_some() {
        BufferedContext::Linear { input: input.clone() }
    } else if layer.as_relu().is_some() {
        BufferedContext::ReLU { input: input.clone() }
    } else if layer.as_sigmoid().is_some() {
        BufferedContext::Sigmoid { output: output.clone() }
    } else if layer.as_tanh().is_some() {
        BufferedContext::Tanh { output: output.clone() }
    } else if layer.as_softmax().is_some() {
        BufferedContext::Softmax { output: output.clone() }
    } else if layer.as_leaky_relu().is_some() {
        BufferedContext::LeakyReLU { input: input.clone() }
    } else if layer.as_identity().is_some() {
        BufferedContext::Identity { input: input.clone() }
    } else if layer.as_memory().is_some() {
        BufferedContext::Memory { input: input.clone() }
    } else if layer.as_soft_sparse_gate().is_some() {
        BufferedContext::SoftSparseGate { input: input.clone() }
    } else if layer.as_soft_keep_gate().is_some() {
        BufferedContext::SoftKeepGate { input: input.clone() }
    } else if layer.as_dual_anchor().is_some() {
        BufferedContext::DualAnchor1D { input: input.clone() }
    } else if layer.as_adaptive_activation().is_some() {
        BufferedContext::AdaptiveActivation { input: input.clone() }
    } else if layer.as_dual_slope_relu().is_some() {
        BufferedContext::DualSlopeReLU { input: input.clone() }
    } else if layer.as_learnable_mish().is_some() {
        BufferedContext::LearnableMish { input: input.clone() }
    } else if layer.as_learnable_softplus().is_some() {
        BufferedContext::LearnableSoftplus { input: input.clone() }
    } else if layer.as_rms_norm_learnable_eps().is_some() {
        BufferedContext::RMSNormWithLearnableEpsilon { input: input.clone() }
    } else if layer.as_adaptive_dropout().is_some() {
        let empty_mask = pool_guard.acquire(0, 0);
        let empty_arg = pool_guard.acquire(0, 0);
        BufferedContext::AdaptiveDropout {
            input: input.clone(),
            mask: empty_mask,
            arg: empty_arg,
        }
    } else if layer.as_feature_fusion().is_some() {
        BufferedContext::FeatureFusion { input: input.clone() }
    } else if layer.as_sparse_feature_selection_gate().is_some() {
        BufferedContext::SparseFeatureSelectionGate { input: input.clone() }
    } else if layer.as_multi_resolution_kan_linear().is_some() {
        BufferedContext::MultiResolutionKANLinear { input: input.clone() }
    } else if layer.as_adaptive_normalization().is_some() {
        BufferedContext::AdaptiveNormalization { input: input.clone() }
    } else if layer.as_batch_renorm().is_some() {
        BufferedContext::BatchRenorm {
            input: input.clone(),
            mean: Vec::new(),
            var: Vec::new(),
            use_batch_stats: true,
        }
    } else if layer.as_concrete_dropout().is_some() {
        let empty_arg = pool_guard.acquire(0, 0);
        BufferedContext::ConcreteDropout {
            input: input.clone(),
            arg: empty_arg,
        }
    } else if layer.as_ind_rnn().is_some() {
        let empty_h = pool_guard.acquire(0, 0);
        BufferedContext::IndRNN {
            input: input.clone(),
            h_all: empty_h,
        }
    } else if layer.as_mamba().is_some() {
        let empty_h = pool_guard.acquire(0, 0);
        BufferedContext::Mamba {
            input: input.clone(),
            h_all: empty_h,
        }
    } else if layer.as_spectral_norm_linear().is_some() {
        BufferedContext::SpectralNormLinear { input: input.clone() }
    } else {
        BufferedContext::Identity { input: input.clone() }
    }
}

pub(crate) fn can_parallelize(layers: &[Box<dyn UniversalLayer>]) -> bool {
    !layers.iter().any(|l| {
        l.as_memory().is_some()
            || l.as_ind_rnn().is_some()
            || l.as_mamba().is_some()
            || l.as_concrete_dropout().is_some()
            || l.as_adaptive_dropout().is_some()
            || l.as_batch_renorm().is_some()
    })
}

// ============================================================================
//  Параллельный forward
// ============================================================================

/// Один чанк вместе с его глобальным ID.
#[derive(Clone, Copy)]
struct ChunkMeta {
    chunk_id: usize,
    start: usize,
    size: usize,
    end: usize,
}

/// Параллельный прямой проход.
///
/// Возвращает `(контексты, раскладка)`:
/// * `контексты` — `ctx_storage[i]` содержит контексты слоёв для чанка с глобальным
///   `chunk_id = i`.
/// * `раскладка` — `Vec<(start, size, end)>` в том же глобальном порядке
///   (`раскладка[i]` соответствует `контексты[i]`).
///
/// Эту пару обязательно нужно сохранить между forward и backward: scheduler
/// может изменить `plan_chunks_assignment` между вызовами (после обучения
/// mini-model), поэтому backward не должен перезапрашивать раскладку — он
/// получает её как аргумент.
///
/// Состояние чанков отслеживается событийно: воркеры фиксируют переходы
/// `Assigned → InProgress → Done` в `ChunkTracker` в моменты самих переходов.
/// Никаких фоновых таймеров и сторожевых потоков не создаётся.
pub(crate) fn forward_universal_parallel(
    executor: &dyn Executor,
    pool: Arc<Mutex<TempMatrixPool>>,
    layers: Arc<Vec<Box<dyn UniversalLayer>>>,
    slices: Vec<ParamSlice>,
    params: MatrixBufferHandle,
    input: MatrixBufferHandle,
    output: MatrixBufferHandle,
) -> (ChunkedContexts, Vec<(usize, usize, usize)>) {
    let batch_size = input.rows();
    let num_workers = executor.num_workers();

    assert!(
        num_workers >= 1,
        "forward_universal_parallel: worker pool has no workers (num_workers = {})",
        num_workers
    );

    let assignments = executor.plan_chunks_assignment(batch_size);

    assert_eq!(
        assignments.len(),
        num_workers,
        "forward_universal_parallel: scheduler returned {} worker assignments, \
         but pool has {} workers",
        assignments.len(),
        num_workers
    );

    // Нумеруем чанки глобально в том порядке, в каком они пришли.
    let mut global_chunks: Vec<(usize, usize, usize)> = Vec::new();
    let mut per_worker_chunks: Vec<Vec<ChunkMeta>> = Vec::with_capacity(num_workers);

    for worker_chunks in assignments {
        let mut metas = Vec::with_capacity(worker_chunks.len());
        for (start, size, end) in worker_chunks {
            let cid = global_chunks.len();
            global_chunks.push((start, size, end));
            metas.push(ChunkMeta { chunk_id: cid, start, size, end });
        }
        per_worker_chunks.push(metas);
    }

    let total_chunks = global_chunks.len();
    if total_chunks == 0 {
        return (Vec::new(), Vec::new());
    }

    // Трекер заполняется событийно. Никаких потоков, никаких таймеров.
    let tracker = Arc::new(Mutex::new(ChunkTracker::new(&global_chunks)));
    {
        let mut t = tracker.lock().unwrap();
        for (logical_worker_id, metas) in per_worker_chunks.iter().enumerate() {
            for m in metas {
                t.mark_assigned(m.chunk_id, logical_worker_id);
            }
        }
    }

    let slices_arc = Arc::new(slices);
    let shared = Arc::new(ForwardTaskShared {
        input,
        output,
        params,
        layers,
        slices: slices_arc,
        pool,
    });

    let ctx_storage: Arc<Mutex<Vec<Vec<DynamicContext>>>> =
        Arc::new(Mutex::new(vec![Vec::new(); total_chunks]));

    // Владеющий `'static`-хэндл для воркеров.
    let executor_arc: Arc<dyn Executor> = Arc::from(executor.clone_executor());

    for (logical_worker_id, metas) in per_worker_chunks.into_iter().enumerate() {
        let shared = shared.clone();
        let tracker = tracker.clone();
        let ctx_storage = ctx_storage.clone();
        let executor_for_worker = Arc::clone(&executor_arc);

        let task = Box::new(move || {
            let physical_worker_id = WorkerPool::current_worker_index();

            let mut pool_guard = shared.pool.lock().unwrap();

            for m in metas {
                tracker
                    .lock()
                    .unwrap()
                    .mark_in_progress(m.chunk_id, physical_worker_id);

                let t0 = Instant::now();

                let input_chunk =
                    extract_chunk(&shared.input, m.start, m.end, &mut *pool_guard);
                let mut current = input_chunk;
                let mut chunk_ctxs = Vec::with_capacity(shared.layers.len());

                for (layer, slice) in shared.layers.iter().zip(shared.slices.iter()) {
                    let out_cols = get_output_features(layer, &current);
                    let out = pool_guard.acquire(current.rows(), out_cols);
                    let buffered_ctx =
                        build_buffered_context(layer, &current, &out, &mut *pool_guard);
                    call_forward_buffered(layer, &current, &out, &shared.params, slice);
                    chunk_ctxs.push(DynamicContext::Buffered(buffered_ctx));
                    current = out;
                }

                write_chunk(&shared.output, &current, m.start);

                {
                    let mut storage = ctx_storage.lock().unwrap();
                    storage[m.chunk_id] = chunk_ctxs;
                }

                let duration_ns = t0.elapsed().as_nanos() as u64;

                tracker
                    .lock()
                    .unwrap()
                    .mark_done(m.chunk_id, duration_ns);

                // Обратная связь scheduler'у — обучение mini-model.
                executor_for_worker.report_execution_time(
                    logical_worker_id,
                    m.size,
                    duration_ns as f64,
                );
            }
        });

        executor.execute_dyn(task);
    }

    executor.wait_all();

    let storage = ctx_storage.lock().unwrap();
    (storage.clone(), global_chunks)
}

// ============================================================================
//  Параллельный backward
// ============================================================================

/// Параллельный обратный проход.
///
/// Раскладку чанков **не запрашивает заново** у scheduler'а — она приходит
/// аргументом `saved_chunks` из forward-прохода. Это гарантирует, что
/// `contexts[i]` (сохранённый forward-контекст) соответствует `saved_chunks[i]`
/// (диапазон батча).
///
/// Чанки распределяются по воркерам простым round-robin: воркер `k`
/// обрабатывает чанки `k, k + num_workers, k + 2·num_workers, …`.
/// Если `saved_chunks.len() < num_workers` — часть воркеров получит пустой
/// набор задач и ничего не будет делать (это корректно).
///
/// Как и в forward, состояние чанков отслеживается событийно, без таймеров.
pub(crate) fn backward_universal_parallel(
    executor: &dyn Executor,
    pool: Arc<Mutex<TempMatrixPool>>,
    layers: Arc<Vec<Box<dyn UniversalLayer>>>,
    slices: Vec<ParamSlice>,
    contexts: ChunkedContexts,
    saved_chunks: &[(usize, usize, usize)],
    grad_output: MatrixBufferHandle,
    grad_input: MatrixBufferHandle,
    params: MatrixBufferHandle,
    grad_params: MatrixBufferHandle,
) {
    let batch_size = grad_output.rows();
    let num_workers = executor.num_workers();

    assert!(
        num_workers >= 1,
        "backward_universal_parallel: worker pool has no workers (num_workers = {})",
        num_workers
    );

    let total_chunks = saved_chunks.len();
    if total_chunks == 0 {
        return;
    }

    assert_eq!(
        total_chunks,
        contexts.len(),
        "backward_universal_parallel: number of saved chunks ({}) does not match contexts ({})",
        total_chunks,
        contexts.len()
    );

    // Round-robin по воркерам. Воркер k получает чанки k, k+nw, k+2·nw, ...
    let mut per_worker_chunks: Vec<Vec<ChunkMeta>> =
        (0..num_workers).map(|_| Vec::new()).collect();

    for (chunk_id, &(start, size, end)) in saved_chunks.iter().enumerate() {
        let logical_worker_id = chunk_id % num_workers;
        per_worker_chunks[logical_worker_id].push(ChunkMeta {
            chunk_id,
            start,
            size,
            end,
        });
    }

    let _ = batch_size; // сохраняем для отладки при необходимости

    // Трекер заполняется событийно.
    let tracker = Arc::new(Mutex::new(ChunkTracker::new(saved_chunks)));
    {
        let mut t = tracker.lock().unwrap();
        for (logical_worker_id, metas) in per_worker_chunks.iter().enumerate() {
            for m in metas {
                t.mark_assigned(m.chunk_id, logical_worker_id);
            }
        }
    }

    let slices_arc = Arc::new(slices);
    let shared = Arc::new(BackwardTaskShared {
        grad_output,
        grad_input,
        params,
        grad_params,
        layers,
        slices: slices_arc,
        contexts,
        pool,
    });

    // Временные буферы градиентов — по одному на чанк.
    let param_len = shared.grad_params.rows();
    let mut temp_grads: Vec<MatrixBufferHandle> = Vec::with_capacity(total_chunks);
    for _ in 0..total_chunks {
        let h = shared.pool.lock().unwrap().acquire(param_len, 1);
        temp_grads.push(h);
    }
    let temp_grads = Arc::new(temp_grads);

    for metas in per_worker_chunks.into_iter() {
        if metas.is_empty() {
            continue;
        }

        let shared = shared.clone();
        let tracker = tracker.clone();
        let temp_grads = temp_grads.clone();

        let task = Box::new(move || {
            let physical_worker_id = WorkerPool::current_worker_index();

            let mut pool_guard = shared.pool.lock().unwrap();

            for m in metas {
                tracker
                    .lock()
                    .unwrap()
                    .mark_in_progress(m.chunk_id, physical_worker_id);

                let t0 = Instant::now();

                let grad_output_chunk =
                    extract_chunk(&shared.grad_output, m.start, m.end, &mut *pool_guard);
                let mut current_grad = grad_output_chunk;

                let contexts_chunk = &shared.contexts[m.chunk_id];
                let temp_grad = &temp_grads[m.chunk_id];

                for i in (0..shared.layers.len()).rev() {
                    let layer = &shared.layers[i];
                    let slice = &shared.slices[i];
                    let ctx = &contexts_chunk[i];

                    let in_features = get_input_features(layer, &current_grad);
                    let grad_input_chunk =
                        pool_guard.acquire(current_grad.rows(), in_features);

                    call_backward_buffered(
                        layer,
                        ctx,
                        &current_grad,
                        &grad_input_chunk,
                        &shared.params,
                        slice,
                        temp_grad,
                    );

                    pool_guard.release(current_grad);
                    current_grad = grad_input_chunk;
                }

                write_chunk(&shared.grad_input, &current_grad, m.start);
                pool_guard.release(current_grad);

                let duration_ns = t0.elapsed().as_nanos() as u64;

                tracker
                    .lock()
                    .unwrap()
                    .mark_done(m.chunk_id, duration_ns);
            }
        });

        executor.execute_dyn(task);
    }

    executor.wait_all();

    // Финальная редукция: суммируем temp_grads в grad_params.
    let mut pool_guard = shared.pool.lock().unwrap();
    {
        let mut grad_guard = shared.grad_params.write();
        let grad_slice = grad_guard.as_slice_mut().expect("CPU buffer");
        for v in grad_slice.iter_mut() {
            *v = 0.0;
        }
    }
    for temp in temp_grads.iter() {
        let temp_guard = temp.read();
        let temp_slice = temp_guard.as_slice().expect("CPU buffer");
        let mut grad_guard = shared.grad_params.write();
        let grad_slice = grad_guard.as_slice_mut().expect("CPU buffer");
        for i in 0..grad_slice.len() {
            grad_slice[i] += temp_slice[i];
        }
    }
    for temp in temp_grads.iter() {
        pool_guard.release(temp.clone());
    }
}