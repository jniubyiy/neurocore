// src/compute_manager/graph_v2/forward_v2.rs
//
// Forward-проход GraphV2.
//
// Модуль принимает всё нужное явно: distributor, param_store, segments,
// input handle. Граф (model_v2) — тонкий фасад, вызывающий этот модуль.
//
// Модель потоков:
//
//   * Граф работает с ВЕКТОРОМ потоков. Обычно длина = 1, но после
//     Splitter — 2, после Combiner — снова 1.
//
//   * Universal / DimOp работают с одним конкретным потоком по индексу
//     `seg.stream_indices`. Если `stream_indices = None` — используется
//     `streams[0]`. Результат работы кладётся на ту же позицию:
//     `streams[idx] = output`.
//
//   * Splitter принимает ровно 1 поток (`streams[0]`) и заменяет весь
//     вектор на `[out_a, out_b]`.
//
//   * Combiner принимает ровно 2 потока (`streams[0], streams[1]`) и
//     заменяет весь вектор на `[out]`.
//
// # Кэш (MIGRATION_PLAN.md §7, Фаза 4)
//
// `run_forward` сохраняет `batch` (число строк входного буфера) в
// `ForwardCacheV2`. Это значение переживёт backward и будет использовано
// `adapter_pass` для заполнения `AdapterContext::batch`.

use std::sync::{Arc, Mutex};

use crate::compute_manager::distributor_v2::SmartDistributor;
use crate::compute_manager::jobs_v2::{
    ConnectorDirection, ConnectorOpJob, ConnectorOpKind, DimOpJob, ForwardSegmentJob,
    Job, JobResult,
};
use crate::compute_manager::operators_v2::memory_v2::buffer::MatrixBufferHandle;
use crate::model_plan::param_store::ParamStore;

use super::types_v2::{
    ForwardCacheV2, SegmentForwardStateV2, SegmentKindV2, SegmentV2,
};

/// Выполняет forward-проход через граф.
///
/// Возвращает `(output_handle, cache)`. Handle живёт в temp-pool
/// распределителя; освобождается автоматически при drop'е.
pub(super) fn run_forward(
    distributor: &SmartDistributor,
    param_store: &Arc<Mutex<ParamStore>>,
    segments: &[SegmentV2],
    input: MatrixBufferHandle,
) -> Result<(MatrixBufferHandle, ForwardCacheV2), String> {
    // Запоминаем batch до move `input` в `streams`.
    // (MIGRATION_PLAN.md §7, Фаза 4: batch сохраняется в ForwardCacheV2.)
    let batch = input.rows();

    let mut streams: Vec<MatrixBufferHandle> = vec![input];
    let mut states: Vec<SegmentForwardStateV2> = Vec::with_capacity(segments.len());

    for seg in segments {
        let (next_streams, state) =
            run_segment_forward(distributor, param_store, seg, streams)?;
        streams = next_streams;
        states.push(state);
    }

    if streams.len() != 1 {
        return Err(format!(
            "GraphV2::forward: expected single output stream, got {}",
            streams.len()
        ));
    }

    let output = streams.into_iter().next().unwrap();
    let cache = ForwardCacheV2 {
        segment_states: states,
        output: output.clone(),
        batch,
    };

    Ok((output, cache))
}

// ============================================================================
// Один сегмент
// ============================================================================

/// Возвращает одиночный индекс потока для Universal/DimOp.
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
                "GraphV2::forward: segment {} ({}): stream_indices must be a single index, got {:?}",
                seg_index, kind_name, other
            ));
        }
    };
    if idx >= streams_len {
        return Err(format!(
            "GraphV2::forward: segment {} ({}): stream index {} out of range ({} streams)",
            seg_index, kind_name, idx, streams_len
        ));
    }
    Ok(idx)
}

fn run_segment_forward(
    distributor: &SmartDistributor,
    param_store: &Arc<Mutex<ParamStore>>,
    seg: &SegmentV2,
    streams: Vec<MatrixBufferHandle>,
) -> Result<(Vec<MatrixBufferHandle>, SegmentForwardStateV2), String> {
    match &seg.kind {
        SegmentKindV2::Universal { layers, slices } => {
            let idx = single_stream_index(
                seg.index,
                seg.stream_indices.as_deref(),
                streams.len(),
                "Universal",
            )?;

            let params_handle = {
                let ps = param_store.lock().unwrap();
                let first_slice = slices.first().ok_or_else(|| {
                    format!("GraphV2::forward: segment {} has no slices", seg.index)
                })?;
                ps.params_handle(first_slice).clone()
            };

            let input = streams[idx].clone();

            let job = Job::ForwardSegment(ForwardSegmentJob {
                segment_index: seg.index,
                layers: Arc::clone(layers),
                slices: slices.clone(),
                params: params_handle,
                input,
            });

            match distributor.dispatch(job) {
                JobResult::Forward { output, contexts } => {
                    let mut new_streams = streams;
                    new_streams[idx] = output;
                    Ok((
                        new_streams,
                        SegmentForwardStateV2::Universal { contexts },
                    ))
                }
                JobResult::Failed(msg) => Err(format!(
                    "GraphV2::forward: segment {} (Universal): {}",
                    seg.index, msg
                )),
                _ => Err(format!(
                    "GraphV2::forward: segment {} (Universal): unexpected JobResult",
                    seg.index
                )),
            }
        }

        SegmentKindV2::DimOp { kind } => {
            let idx = single_stream_index(
                seg.index,
                seg.stream_indices.as_deref(),
                streams.len(),
                "DimOp",
            )?;

            let input = streams[idx].clone();

            let job = Job::DimOp(DimOpJob {
                kind: kind.clone(),
                input,
            });
            match distributor.dispatch(job) {
                JobResult::Buffer(out) => {
                    let mut new_streams = streams;
                    new_streams[idx] = out;
                    Ok((new_streams, SegmentForwardStateV2::None))
                }
                JobResult::Failed(msg) => Err(format!(
                    "GraphV2::forward: segment {} (DimOp): {}",
                    seg.index, msg
                )),
                _ => Err(format!(
                    "GraphV2::forward: segment {} (DimOp): unexpected JobResult",
                    seg.index
                )),
            }
        }

        SegmentKindV2::Connector { kind } => {
            run_connector_forward(distributor, param_store, seg, kind, streams)
        }
    }
}

fn run_connector_forward(
    distributor: &SmartDistributor,
    param_store: &Arc<Mutex<ParamStore>>,
    seg: &SegmentV2,
    kind: &ConnectorOpKind,
    streams: Vec<MatrixBufferHandle>,
) -> Result<(Vec<MatrixBufferHandle>, SegmentForwardStateV2), String> {
    match kind {
        ConnectorOpKind::Splitter { slice, .. } => {
            if streams.len() != 1 {
                return Err(format!(
                    "GraphV2::forward: segment {} (Splitter): expected single input stream, got {}",
                    seg.index,
                    streams.len()
                ));
            }
            let params_handle = {
                let ps = param_store.lock().unwrap();
                ps.params_handle(slice).clone()
            };

            let x = streams.into_iter().next().unwrap();

            let job = Job::ConnectorOp(ConnectorOpJob {
                direction: ConnectorDirection::Forward,
                kind: kind.clone(),
                inputs: vec![x.clone()],
                params: Some(params_handle),
                grad_params: None,
                saved: Vec::new(),
            });
            match distributor.dispatch(job) {
                JobResult::Buffers(mut outs) => {
                    if outs.len() != 4 {
                        return Err(format!(
                            "GraphV2::forward: segment {} (Splitter): expected 4 buffers \
                             (out_a, out_b, pre_a, pre_b), got {}",
                            seg.index,
                            outs.len()
                        ));
                    }
                    let pre_b = outs.pop().unwrap();
                    let pre_a = outs.pop().unwrap();
                    let out_b = outs.pop().unwrap();
                    let out_a = outs.pop().unwrap();
                    let state = SegmentForwardStateV2::Connector {
                        inputs: vec![x],
                        pre: vec![pre_a, pre_b],
                    };
                    Ok((vec![out_a, out_b], state))
                }
                JobResult::Failed(msg) => Err(format!(
                    "GraphV2::forward: segment {} (Splitter): {}",
                    seg.index, msg
                )),
                _ => Err(format!(
                    "GraphV2::forward: segment {} (Splitter): unexpected JobResult",
                    seg.index
                )),
            }
        }

        ConnectorOpKind::Combiner { slice, .. } => {
            if streams.len() != 2 {
                return Err(format!(
                    "GraphV2::forward: segment {} (Combiner): expected 2 input streams, got {}",
                    seg.index,
                    streams.len()
                ));
            }
            let params_handle = {
                let ps = param_store.lock().unwrap();
                ps.params_handle(slice).clone()
            };

            let inputs: Vec<MatrixBufferHandle> = streams.clone();

            let job = Job::ConnectorOp(ConnectorOpJob {
                direction: ConnectorDirection::Forward,
                kind: kind.clone(),
                inputs: inputs.clone(),
                params: Some(params_handle),
                grad_params: None,
                saved: Vec::new(),
            });
            match distributor.dispatch(job) {
                JobResult::Buffers(mut outs) => {
                    if outs.len() != 2 {
                        return Err(format!(
                            "GraphV2::forward: segment {} (Combiner): expected 2 buffers \
                             (out, pre), got {}",
                            seg.index,
                            outs.len()
                        ));
                    }
                    let pre = outs.pop().unwrap();
                    let out = outs.pop().unwrap();
                    let state = SegmentForwardStateV2::Connector {
                        inputs,
                        pre: vec![pre],
                    };
                    Ok((vec![out], state))
                }
                JobResult::Failed(msg) => Err(format!(
                    "GraphV2::forward: segment {} (Combiner): {}",
                    seg.index, msg
                )),
                _ => Err(format!(
                    "GraphV2::forward: segment {} (Combiner): unexpected JobResult",
                    seg.index
                )),
            }
        }
    }
}