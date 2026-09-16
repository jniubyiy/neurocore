// src/compute_manager/cpu/parallel.rs
//
// Параллельный forward/backward по чанкам батча.
//
// Ключевая идея forward'а:
//   * Планировщик разбивает батч на чанки (start, size, end).
//   * Для каждого чанка строится ChunkSlice — описание того, какие
//     строки входа он читает и в какие строки выхода он пишет свой
//     результат. В стандартном режиме in_range == out_range.
//   * Каждый воркер для каждого своего чанка:
//       - извлекает входной срез из общего input;
//       - прогоняет через всю цепочку слоёв;
//       - получает BufferedContext от каждого слоя (слой сам строит
//         свой контекст, включая per-chunk state, если он есть);
//       - пишет финальный результат в СВОЙ per-chunk буфер;
//       - сохраняет контексты слоёв в ctx_storage[chunk_id].
//   * После завершения всех воркеров выполняется фаза merge:
//     последовательно копирует per-chunk буферы в output по
//     out_start..out_end из плана.
//
// Backward устроен проще: градиенты пишутся в общий grad_input сразу
// по диапазону входного среза, потому что направление потока
// градиента однозначно определено forward'ом.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use once_cell::sync::Lazy;

use crate::compute_manager::cpu::WorkerPool;
use crate::compute_manager::executor::Executor;
use crate::compute_manager::graph::types::{ChunkedContexts, DynamicContext};
use crate::compute_manager::matrix_buffer::{MatrixBufferHandle, TempMatrixPool};
use crate::layers::{
    UniversalLayer, UniversalLayerBuffered,
    Linear, ReLU, Sigmoid, Tanh, LeakyReLU, Identity, Softmax,
    Memory, SoftSparseGate, SoftKeepGate, DualAnchor, AdaptivePerFeatureActivation,
    DualSlopeReLU, LearnableMish, LearnableSoftplus, RMSNormWithLearnableEpsilon,
    AdaptiveDropout, FeatureFusion, SparseFeatureSelectionGate, MultiResolutionKANLinear,
    AdaptiveNormalization, BatchRenorm1d, ConcreteDropout, IndRNN, Mamba,
    SpectrallyNormalizedLinear, LinearAttention, RelativePositionAttention,
    BufferedContext,
};
use crate::model_plan::param_store::ParamSlice;

// ============================================================================
//  Отладочные переключатели
// ============================================================================

static PARALLEL_DEBUG: Lazy<bool> =
    Lazy::new(|| std::env::var("NEUROCORE_DEBUG_PARALLEL").is_ok());

// ============================================================================
//  План чанкования
// ============================================================================

/// Описание одного чанка в плане forward.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ChunkSlice {
    pub chunk_id: usize,
    pub worker_id: usize,
    pub in_start: usize,
    pub in_end: usize,
    pub out_start: usize,
    pub out_end: usize,
}

impl ChunkSlice {
    #[inline]
    pub fn in_size(&self) -> usize {
        self.in_end - self.in_start
    }

    #[inline]
    pub fn as_layout_tuple(&self) -> (usize, usize, usize) {
        (self.in_start, self.in_size(), self.in_end)
    }
}

/// План распределения forward по чанкам.
#[derive(Clone, Debug)]
pub(crate) struct LayerChunkPlan {
    pub chunks: Vec<ChunkSlice>,
}

impl LayerChunkPlan {
    pub fn to_layout(&self) -> Vec<(usize, usize, usize)> {
        self.chunks.iter().map(|c| c.as_layout_tuple()).collect()
    }
}

/// Строит стандартный план чанкования из раскладки планировщика.
pub(crate) fn default_layer_chunk_plan(
    assignments: &[Vec<(usize, usize, usize)>],
) -> LayerChunkPlan {
    let mut chunks = Vec::new();
    let mut chunk_id = 0usize;
    for (worker_id, worker_chunks) in assignments.iter().enumerate() {
        for &(start, _size, end) in worker_chunks {
            chunks.push(ChunkSlice {
                chunk_id,
                worker_id,
                in_start: start,
                in_end: end,
                out_start: start,
                out_end: end,
            });
            chunk_id += 1;
        }
    }
    LayerChunkPlan { chunks }
}

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
                c.chunk_id, c.start, c.end, c.assigned_worker, c.physical_worker,
                c.duration_ns as f64 / 1000.0
            );
        }
        if !any { let _ = writeln!(s, "  (none)"); }

        let _ = writeln!(s, "--- in_progress ---");
        any = false;
        for c in self.chunks.iter().filter(|c| c.status == ChunkStatus::InProgress) {
            any = true;
            let running = c.started_at
                .map(|a| format!("{:.3}s", a.elapsed().as_secs_f64()))
                .unwrap_or_else(|| "?".into());
            let _ = writeln!(
                s,
                "  chunk {:>3}  range [{:>3}..{:<3})  assigned={:?}  physical={:?}  running={}",
                c.chunk_id, c.start, c.end, c.assigned_worker, c.physical_worker, running
            );
        }
        if !any { let _ = writeln!(s, "  (none)"); }

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
        if !any { let _ = writeln!(s, "  (none)"); }

        let _ = writeln!(s, "--- pending ---");
        any = false;
        for c in self.chunks.iter().filter(|c| c.status == ChunkStatus::Pending) {
            any = true;
            let _ = writeln!(s, "  chunk {:>3}  range [{:>3}..{:<3})", c.chunk_id, c.start, c.end);
        }
        if !any { let _ = writeln!(s, "  (none)"); }

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

/// Записывает чанк в выходной буфер, начиная со строки `start`.
pub(crate) fn write_chunk(
    output: &MatrixBufferHandle,
    chunk: &MatrixBufferHandle,
    start: usize,
) {
    let out_rows = output.rows();
    let cols = output.cols();
    let chunk_rows = chunk.rows();

    if cols != chunk.cols() {
        eprintln!(
            "[WRITE-CHUNK-MISMATCH] output={}x{} chunk={}x{} start={} \
             (cols mismatch: output.cols()={} chunk.cols()={})",
            out_rows, cols, chunk_rows, chunk.cols(), start, cols, chunk.cols(),
        );
    }

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

pub(crate) fn write_chunk_to_range(
    output: &MatrixBufferHandle,
    chunk: &MatrixBufferHandle,
    out_start: usize,
    out_end: usize,
) {
    let chunk_rows = chunk.rows();
    assert_eq!(
        out_end - out_start,
        chunk_rows,
        "write_chunk_to_range: range size ({}) must match chunk rows ({})",
        out_end - out_start,
        chunk_rows,
    );
    write_chunk(output, chunk, out_start);
}

// ============================================================================
//  Размерности слоёв
// ============================================================================

#[inline]
fn get_output_features(layer: &Box<dyn UniversalLayer>, input: &MatrixBufferHandle) -> usize {
    layer.output_features_for(input.cols())
}

#[inline]
fn get_input_features(layer: &Box<dyn UniversalLayer>, grad_output: &MatrixBufferHandle) -> usize {
    layer.input_features_for(grad_output.cols())
}

// ============================================================================
//  Диспетчеризация слоёв
// ============================================================================

/// Единая точка вызова forward слоя. Каждый слой сам строит свой
/// `BufferedContext` — включая per-chunk state, если он есть.
fn call_forward_buffered(
    layer: &Box<dyn UniversalLayer>,
    input: &MatrixBufferHandle,
    output: &MatrixBufferHandle,
    params: &MatrixBufferHandle,
    slice: &ParamSlice,
    pool: &mut TempMatrixPool,
) -> BufferedContext {
    if let Some(l) = layer.as_linear() {
        <Linear as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice, pool)
    } else if let Some(l) = layer.as_relu() {
        <ReLU as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice, pool)
    } else if let Some(l) = layer.as_sigmoid() {
        <Sigmoid as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice, pool)
    } else if let Some(l) = layer.as_tanh() {
        <Tanh as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice, pool)
    } else if let Some(l) = layer.as_leaky_relu() {
        <LeakyReLU as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice, pool)
    } else if let Some(l) = layer.as_identity() {
        <Identity as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice, pool)
    } else if let Some(l) = layer.as_softmax() {
        <Softmax as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice, pool)
    } else if let Some(l) = layer.as_memory() {
        <Memory as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice, pool)
    } else if let Some(l) = layer.as_soft_sparse_gate() {
        <SoftSparseGate as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice, pool)
    } else if let Some(l) = layer.as_soft_keep_gate() {
        <SoftKeepGate as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice, pool)
    } else if let Some(l) = layer.as_dual_anchor() {
        <DualAnchor as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice, pool)
    } else if let Some(l) = layer.as_adaptive_activation() {
        <AdaptivePerFeatureActivation as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice, pool)
    } else if let Some(l) = layer.as_dual_slope_relu() {
        <DualSlopeReLU as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice, pool)
    } else if let Some(l) = layer.as_learnable_mish() {
        <LearnableMish as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice, pool)
    } else if let Some(l) = layer.as_learnable_softplus() {
        <LearnableSoftplus as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice, pool)
    } else if let Some(l) = layer.as_rms_norm_learnable_eps() {
        <RMSNormWithLearnableEpsilon as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice, pool)
    } else if let Some(l) = layer.as_adaptive_dropout() {
        <AdaptiveDropout as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice, pool)
    } else if let Some(l) = layer.as_feature_fusion() {
        <FeatureFusion as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice, pool)
    } else if let Some(l) = layer.as_sparse_feature_selection_gate() {
        <SparseFeatureSelectionGate as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice, pool)
    } else if let Some(l) = layer.as_multi_resolution_kan_linear() {
        <MultiResolutionKANLinear as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice, pool)
    } else if let Some(l) = layer.as_adaptive_normalization() {
        <AdaptiveNormalization as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice, pool)
    } else if let Some(l) = layer.as_batch_renorm() {
        <BatchRenorm1d as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice, pool)
    } else if let Some(l) = layer.as_concrete_dropout() {
        <ConcreteDropout as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice, pool)
    } else if let Some(l) = layer.as_ind_rnn() {
        <IndRNN as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice, pool)
    } else if let Some(l) = layer.as_mamba() {
        <Mamba as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice, pool)
    } else if let Some(l) = layer.as_spectral_norm_linear() {
        <SpectrallyNormalizedLinear as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice, pool)
    } else if let Some(l) = layer.as_linear_attention() {
        <LinearAttention as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice, pool)
    } else if let Some(l) = layer.as_relative_position_attention() {
        <RelativePositionAttention as UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice, pool)
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
    } else if let Some(l) = layer.as_linear_attention() {
        <LinearAttention as UniversalLayerBuffered>::backward_buffered(l, ctx, grad_output, grad_input, params, slice, grad_params);
    } else if let Some(l) = layer.as_relative_position_attention() {
        <RelativePositionAttention as UniversalLayerBuffered>::backward_buffered(l, ctx, grad_output, grad_input, params, slice, grad_params);
    } else {
        unreachable!("Unsupported layer in parallel backward");
    }
}

/// Определяет, можно ли распараллелить цепочку слоёв по чанкам батча.
///
/// Список несовместимых слоёв сохранён как временный fallback; в дальнейшем
/// он будет заменён на декларативный флаг `supports_chunked_parallel()`.
pub(crate) fn can_parallelize(layers: &[Box<dyn UniversalLayer>]) -> bool {
    !layers.iter().any(|l| {
        l.as_memory().is_some()
            || l.as_ind_rnn().is_some()
            || l.as_mamba().is_some()
            || l.as_concrete_dropout().is_some()
            || l.as_adaptive_dropout().is_some()
            || l.as_batch_renorm().is_some()
            || l.as_linear_attention().is_some()
            || l.as_relative_position_attention().is_some()
    })
}

// ============================================================================
//  Параллельный forward
// ============================================================================

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

    let plan = default_layer_chunk_plan(&assignments);
    let total_chunks = plan.chunks.len();

    if total_chunks == 0 {
        return (Vec::new(), Vec::new());
    }

    let mut per_worker_chunks: Vec<Vec<ChunkSlice>> =
        (0..num_workers).map(|_| Vec::new()).collect();
    for chunk in &plan.chunks {
        per_worker_chunks[chunk.worker_id].push(*chunk);
    }

    if *PARALLEL_DEBUG {
        eprintln!(
            "[FWD-PARALLEL-START] batch_size={} num_workers={} total_chunks={} \
             input=({}x{}) output=({}x{}) layers={}",
            batch_size, num_workers, total_chunks,
            input.rows(), input.cols(),
            output.rows(), output.cols(),
            layers.len(),
        );
    }

    let layout_for_tracker: Vec<(usize, usize, usize)> = plan.to_layout();
    let tracker = Arc::new(Mutex::new(ChunkTracker::new(&layout_for_tracker)));
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

    let chunk_outputs: Arc<Mutex<Vec<Option<MatrixBufferHandle>>>> =
        Arc::new(Mutex::new((0..total_chunks).map(|_| None).collect()));

    let executor_arc: Arc<dyn Executor> = Arc::from(executor.clone_executor());

    for (logical_worker_id, metas) in per_worker_chunks.into_iter().enumerate() {
        if metas.is_empty() {
            continue;
        }
        let shared = shared.clone();
        let tracker = tracker.clone();
        let ctx_storage = ctx_storage.clone();
        let chunk_outputs = chunk_outputs.clone();
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
                    extract_chunk(&shared.input, m.in_start, m.in_end, &mut *pool_guard);
                let mut current = input_chunk;
                let mut chunk_ctxs = Vec::with_capacity(shared.layers.len());

                for (layer, slice) in shared.layers.iter().zip(shared.slices.iter()) {
                    let out_cols = get_output_features(layer, &current);
                    let out = pool_guard.acquire(current.rows(), out_cols);

                    if *PARALLEL_DEBUG {
                        eprintln!(
                            "[FWD-PARALLEL] worker={} chunk={} layer={} in=({}x{}) out_cols={}",
                            physical_worker_id,
                            m.chunk_id,
                            std::any::type_name_of_val(layer.as_ref()),
                            current.rows(),
                            current.cols(),
                            out_cols,
                        );
                    }

                    // Слой сам строит свой BufferedContext — включая
                    // per-chunk state, если он есть.
                    let buffered_ctx = call_forward_buffered(
                        layer,
                        &current,
                        &out,
                        &shared.params,
                        slice,
                        &mut *pool_guard,
                    );

                    if *PARALLEL_DEBUG {
                        eprintln!(
                            "[FWD-PARALLEL]   after forward: out=({}x{})",
                            out.rows(),
                            out.cols(),
                        );
                    }

                    chunk_ctxs.push(DynamicContext::Buffered(buffered_ctx));
                    current = out;
                }

                {
                    let mut outputs = chunk_outputs.lock().unwrap();
                    outputs[m.chunk_id] = Some(current);
                }

                {
                    let mut storage = ctx_storage.lock().unwrap();
                    storage[m.chunk_id] = chunk_ctxs;
                }

                let duration_ns = t0.elapsed().as_nanos() as u64;

                tracker
                    .lock()
                    .unwrap()
                    .mark_done(m.chunk_id, duration_ns);

                executor_for_worker.report_execution_time(
                    logical_worker_id,
                    m.in_size(),
                    duration_ns as f64,
                );
            }
        });

        executor.execute_dyn(task);
    }

    executor.wait_all();

    {
        let mut outputs = chunk_outputs.lock().unwrap();
        let mut pool_guard = shared.pool.lock().unwrap();

        for chunk in &plan.chunks {
            let handle = outputs[chunk.chunk_id]
                .take()
                .expect("forward_universal_parallel: missing chunk output after wait_all");

            write_chunk_to_range(
                &shared.output,
                &handle,
                chunk.out_start,
                chunk.out_end,
            );

            pool_guard.release(handle);
        }
    }

    let storage = ctx_storage.lock().unwrap();
    (storage.clone(), plan.to_layout())
}

// ============================================================================
//  Параллельный backward
// ============================================================================

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

    let mut per_worker_chunks: Vec<Vec<(usize, usize, usize, usize)>> =
        (0..num_workers).map(|_| Vec::new()).collect();

    for (chunk_id, &(start, size, end)) in saved_chunks.iter().enumerate() {
        let logical_worker_id = chunk_id % num_workers;
        per_worker_chunks[logical_worker_id].push((chunk_id, start, size, end));
    }

    let _ = batch_size;

    if *PARALLEL_DEBUG {
        eprintln!(
            "[BWD-PARALLEL-START] batch_size={} num_workers={} total_chunks={} \
             grad_output=({}x{}) grad_input=({}x{}) layers={}",
            batch_size, num_workers, total_chunks,
            grad_output.rows(), grad_output.cols(),
            grad_input.rows(), grad_input.cols(),
            layers.len(),
        );
    }

    let tracker = Arc::new(Mutex::new(ChunkTracker::new(saved_chunks)));
    {
        let mut t = tracker.lock().unwrap();
        for (logical_worker_id, metas) in per_worker_chunks.iter().enumerate() {
            for (chunk_id, start, _size, end) in metas {
                let _ = (start, end);
                t.mark_assigned(*chunk_id, logical_worker_id);
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

            for (chunk_id, start, _size, end) in metas {
                tracker
                    .lock()
                    .unwrap()
                    .mark_in_progress(chunk_id, physical_worker_id);

                let t0 = Instant::now();

                let grad_output_chunk =
                    extract_chunk(&shared.grad_output, start, end, &mut *pool_guard);
                let mut current_grad = grad_output_chunk;

                let contexts_chunk = &shared.contexts[chunk_id];
                let temp_grad = &temp_grads[chunk_id];

                for i in (0..shared.layers.len()).rev() {
                    let layer = &shared.layers[i];
                    let slice = &shared.slices[i];
                    let ctx = &contexts_chunk[i];

                    let in_features = get_input_features(layer, &current_grad);
                    let grad_input_chunk =
                        pool_guard.acquire(current_grad.rows(), in_features);

                    if *PARALLEL_DEBUG {
                        eprintln!(
                            "[BWD-PARALLEL] worker={} chunk={} layer={} \
                             grad_in=({}x{}) grad_out=({}x{})",
                            physical_worker_id,
                            chunk_id,
                            std::any::type_name_of_val(layer.as_ref()),
                            current_grad.rows(),
                            in_features,
                            current_grad.rows(),
                            current_grad.cols(),
                        );
                    }

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

                if *PARALLEL_DEBUG || shared.grad_input.cols() != current_grad.cols() {
                    eprintln!(
                        "[BWD-PARALLEL] worker={} chunk={} BEFORE write_chunk: \
                         grad_input=({}x{}) current=({}x{}) start={} end={}",
                        physical_worker_id,
                        chunk_id,
                        shared.grad_input.rows(),
                        shared.grad_input.cols(),
                        current_grad.rows(),
                        current_grad.cols(),
                        start,
                        end,
                    );
                }

                write_chunk(&shared.grad_input, &current_grad, start);
                pool_guard.release(current_grad);

                let duration_ns = t0.elapsed().as_nanos() as u64;

                tracker
                    .lock()
                    .unwrap()
                    .mark_done(chunk_id, duration_ns);
            }
        });

        executor.execute_dyn(task);
    }

    executor.wait_all();

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