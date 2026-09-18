// src/compute_manager/graph_v2/backward_v2.rs
//
// Backward-проход GraphV2.
//
// Идём по сегментам в обратном порядке. Модель потоков симметрична
// forward'у:
//
//   * Universal / DimOp — работают с одним конкретным потоком по индексу
//     `seg.stream_indices`. `grad_output = streams[idx]`, `grad_input`
//     пишется обратно в `streams[idx]`.
//
//   * Splitter (backward) — сворачивает `[delta_a, delta_b]` в один
//     `grad_input`, т.е. заменяет весь вектор на `[dx]`.
//
//   * Combiner (backward) — разворачивает `[delta]` в `[da, db]`.

use std::sync::{Arc, Mutex};

use crate::compute_manager::distributor_v2::SmartDistributor;
use crate::compute_manager::jobs_v2::{
    BackwardSegmentJob, ConnectorDirection, ConnectorOpJob, ConnectorOpKind, DimOpJob,
    DimOpKind, Job, JobResult,
};
use crate::compute_manager::operators_v2::memory_v2::buffer::MatrixBufferHandle;
use crate::model_plan::param_store::ParamStore;

use super::types_v2::{
    ForwardCacheV2, SegmentForwardStateV2, SegmentKindV2, SegmentV2,
};

/// Выполняет backward-проход через граф.
///
/// `grad_output` — градиент по выходу последнего сегмента.
/// Возвращает градиент по входу первого сегмента.
pub(super) fn run_backward(
    distributor: &SmartDistributor,
    param_store: &Arc<Mutex<ParamStore>>,
    segments: &[SegmentV2],
    cache: ForwardCacheV2,
    grad_output: MatrixBufferHandle,
) -> Result<MatrixBufferHandle, String> {
    if cache.segment_states.len() != segments.len() {
        return Err(format!(
            "GraphV2::backward: cache size ({}) != segments count ({})",
            cache.segment_states.len(),
            segments.len()
        ));
    }

    let mut streams: Vec<MatrixBufferHandle> = vec![grad_output];

    for (seg, state) in segments.iter().zip(cache.segment_states.iter()).rev() {
        streams = run_segment_backward(distributor, param_store, seg, state, streams)?;
    }

    if streams.len() != 1 {
        return Err(format!(
            "GraphV2::backward: expected single input gradient, got {}",
            streams.len()
        ));
    }

    Ok(streams.into_iter().next().unwrap())
}

// ============================================================================
// Один сегмент
// ============================================================================

fn single_stream_index(
    seg_index: usize,
    stream_indices: Option<&[usize]>,
    streams_len: usize,
    kind_name: &str,
) -> Result<usize, String> {
    let idx = match stream_indices {
        Some(&[i]) => i,
        None => 0,
        Some(other) => {
            return Err(format!(
                "GraphV2::backward: segment {} ({}): stream_indices must be a single index, got {:?}",
                seg_index, kind_name, other
            ));
        }
    };
    if idx >= streams_len {
        return Err(format!(
            "GraphV2::backward: segment {} ({}): stream index {} out of range ({} streams)",
            seg_index, kind_name, idx, streams_len
        ));
    }
    Ok(idx)
}

fn run_segment_backward(
    distributor: &SmartDistributor,
    param_store: &Arc<Mutex<ParamStore>>,
    seg: &SegmentV2,
    state: &SegmentForwardStateV2,
    streams: Vec<MatrixBufferHandle>,
) -> Result<Vec<MatrixBufferHandle>, String> {
    match (&seg.kind, state) {
        (
            SegmentKindV2::Universal { layers, slices },
            SegmentForwardStateV2::Universal { contexts },
        ) => {
            let idx = single_stream_index(
                seg.index,
                seg.stream_indices.as_deref(),
                streams.len(),
                "Universal",
            )?;

            let (params_handle, grad_params_handle) = {
                let ps = param_store.lock().unwrap();
                let first_slice = slices.first().ok_or_else(|| {
                    format!("GraphV2::backward: segment {} has no slices", seg.index)
                })?;
                (
                    ps.params_handle(first_slice).clone(),
                    ps.grads_handle(first_slice).clone(),
                )
            };

            let grad_output = streams[idx].clone();

            let job = Job::BackwardSegment(BackwardSegmentJob {
                segment_index: seg.index,
                layers: Arc::clone(layers),
                slices: slices.clone(),
                params: params_handle,
                grad_params: grad_params_handle,
                grad_output,
                contexts: contexts.clone(),
            });

            match distributor.dispatch(job) {
                JobResult::Backward { grad_input } => {
                    let mut new_streams = streams;
                    new_streams[idx] = grad_input;
                    Ok(new_streams)
                }
                JobResult::Failed(msg) => Err(format!(
                    "GraphV2::backward: segment {} (Universal): {}",
                    seg.index, msg
                )),
                _ => Err(format!(
                    "GraphV2::backward: segment {} (Universal): unexpected JobResult",
                    seg.index
                )),
            }
        }

        (SegmentKindV2::DimOp { kind }, SegmentForwardStateV2::None) => {
            let idx = single_stream_index(
                seg.index,
                seg.stream_indices.as_deref(),
                streams.len(),
                "DimOp",
            )?;

            // Обратное преобразование: Unsqueeze → ReduceMean, ReduceMean → Unsqueeze.
            let inverse = match kind {
                DimOpKind::Unsqueeze(_) => DimOpKind::ReduceMean(seg.input_shape.clone()),
                DimOpKind::ReduceMean(_) => DimOpKind::Unsqueeze(seg.input_shape.clone()),
            };

            let grad_output = streams[idx].clone();

            let job = Job::DimOp(DimOpJob {
                kind: inverse,
                input: grad_output,
            });
            match distributor.dispatch(job) {
                JobResult::Buffer(out) => {
                    let mut new_streams = streams;
                    new_streams[idx] = out;
                    Ok(new_streams)
                }
                JobResult::Failed(msg) => Err(format!(
                    "GraphV2::backward: segment {} (DimOp inverse): {}",
                    seg.index, msg
                )),
                _ => Err(format!(
                    "GraphV2::backward: segment {} (DimOp inverse): unexpected JobResult",
                    seg.index
                )),
            }
        }

        (
            SegmentKindV2::Connector { kind },
            SegmentForwardStateV2::Connector { inputs, pre },
        ) => run_connector_backward(
            distributor, param_store, seg, kind, inputs, pre, streams,
        ),

        (SegmentKindV2::Connector { .. }, SegmentForwardStateV2::None) => {
            // Connector без forward-состояния — это ошибка: Splitter/Combiner
            // всегда должны сохранять `inputs` и `pre` при forward.
            Err(format!(
                "GraphV2::backward: segment {} (Connector) has no forward state",
                seg.index
            ))
        }

        (_, _) => Err(format!(
            "GraphV2::backward: segment {} state mismatch (kind vs forward state)",
            seg.index
        )),
    }
}

fn run_connector_backward(
    distributor: &SmartDistributor,
    param_store: &Arc<Mutex<ParamStore>>,
    seg: &SegmentV2,
    kind: &ConnectorOpKind,
    forward_inputs: &[MatrixBufferHandle],
    forward_pre: &[MatrixBufferHandle],
    streams: Vec<MatrixBufferHandle>,
) -> Result<Vec<MatrixBufferHandle>, String> {
    match kind {
        ConnectorOpKind::Splitter { slice, .. } => {
            // Backward Splitter: 2 delta → 1 grad_input.
            if streams.len() != 2 {
                return Err(format!(
                    "GraphV2::backward: segment {} (Splitter): expected 2 grad streams, got {}",
                    seg.index,
                    streams.len()
                ));
            }
            if forward_inputs.len() != 1 || forward_pre.len() != 2 {
                return Err(format!(
                    "GraphV2::backward: segment {} (Splitter): bad forward state \
                     (inputs={}, pre={})",
                    seg.index,
                    forward_inputs.len(),
                    forward_pre.len()
                ));
            }

            let (params_handle, grad_params_handle) = {
                let ps = param_store.lock().unwrap();
                (
                    ps.params_handle(slice).clone(),
                    ps.grads_handle(slice).clone(),
                )
            };

            // saved = [x, pre_a, pre_b]
            let saved = vec![
                forward_inputs[0].clone(),
                forward_pre[0].clone(),
                forward_pre[1].clone(),
            ];

            let job = Job::ConnectorOp(ConnectorOpJob {
                direction: ConnectorDirection::Backward,
                kind: kind.clone(),
                inputs: streams,
                params: Some(params_handle),
                grad_params: Some(grad_params_handle),
                saved,
            });
            match distributor.dispatch(job) {
                JobResult::Buffers(outputs) => Ok(outputs),
                JobResult::Failed(msg) => Err(format!(
                    "GraphV2::backward: segment {} (Splitter): {}",
                    seg.index, msg
                )),
                _ => Err(format!(
                    "GraphV2::backward: segment {} (Splitter): unexpected JobResult",
                    seg.index
                )),
            }
        }

        ConnectorOpKind::Combiner { slice, .. } => {
            // Backward Combiner: 1 delta → 2 grad (da, db).
            if streams.len() != 1 {
                return Err(format!(
                    "GraphV2::backward: segment {} (Combiner): expected 1 grad stream, got {}",
                    seg.index,
                    streams.len()
                ));
            }
            if forward_inputs.len() != 2 || forward_pre.len() != 1 {
                return Err(format!(
                    "GraphV2::backward: segment {} (Combiner): bad forward state \
                     (inputs={}, pre={})",
                    seg.index,
                    forward_inputs.len(),
                    forward_pre.len()
                ));
            }

            let (params_handle, grad_params_handle) = {
                let ps = param_store.lock().unwrap();
                (
                    ps.params_handle(slice).clone(),
                    ps.grads_handle(slice).clone(),
                )
            };

            // saved = [a, b, pre]
            let saved = vec![
                forward_inputs[0].clone(),
                forward_inputs[1].clone(),
                forward_pre[0].clone(),
            ];

            let job = Job::ConnectorOp(ConnectorOpJob {
                direction: ConnectorDirection::Backward,
                kind: kind.clone(),
                inputs: streams,
                params: Some(params_handle),
                grad_params: Some(grad_params_handle),
                saved,
            });
            match distributor.dispatch(job) {
                JobResult::Buffers(outputs) => Ok(outputs),
                JobResult::Failed(msg) => Err(format!(
                    "GraphV2::backward: segment {} (Combiner): {}",
                    seg.index, msg
                )),
                _ => Err(format!(
                    "GraphV2::backward: segment {} (Combiner): unexpected JobResult",
                    seg.index
                )),
            }
        }
    }
}