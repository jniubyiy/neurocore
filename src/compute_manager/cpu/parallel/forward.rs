// src/compute_manager/cpu/parallel/forward.rs
//
// Оркестратор параллельного прямого прохода по чанкам батча.

use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::compute_manager::cpu::WorkerPool;
use crate::compute_manager::executor::Executor;
use crate::compute_manager::graph::types::{ChunkedContexts, DynamicContext};
use crate::compute_manager::matrix_buffer::MatrixBufferHandle;
use crate::layers::UniversalLayer;
use crate::model_plan::param_store::ParamSlice;

use super::chunk_ops::{extract_chunk, write_chunk_to_range};
use super::dims::get_output_features;
use super::dispatch::call_forward_buffered;
use super::plan::{default_layer_chunk_plan, ChunkSlice};
use super::shared::{ForwardTaskShared, PARALLEL_DEBUG};
use super::tracker::ChunkTracker;

pub(crate) fn forward_universal_parallel(
    executor: &dyn Executor,
    pool: Arc<Mutex<crate::compute_manager::matrix_buffer::TempMatrixPool>>,
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