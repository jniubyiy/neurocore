// src/compute_manager/cpu/parallel/backward.rs
//
// Оркестратор параллельного обратного прохода по чанкам батча.

use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::compute_manager::operators_v2::cpu_v2::WorkerPool;
use crate::compute_manager::core::executor::Executor;
use crate::compute_manager::core::dynamic_context::ChunkedContexts;
use crate::compute_manager::operators_v2::memory_v2::buffer::MatrixBufferHandle;
use crate::layers::UniversalLayer;
use crate::model_plan::param_store::ParamSlice;

use super::chunk_ops::{extract_chunk, write_chunk};
use super::dims::get_input_features;
use super::dispatch::call_backward_buffered;
use super::shared::BackwardTaskShared;
use super::tracker::ChunkTracker;

#[allow(clippy::too_many_arguments)]
pub(crate) fn backward_universal_parallel(
    executor: &dyn Executor,
    pool: Arc<Mutex<crate::compute_manager::operators_v2::memory_v2::buffer::TempMatrixPool>>,
    layers: Arc<Vec<Box<dyn UniversalLayer>>>,
    slices: Vec<ParamSlice>,
    contexts: ChunkedContexts,
    saved_chunks: &[(usize, usize, usize)],
    grad_output: MatrixBufferHandle,
    grad_input: MatrixBufferHandle,
    params: MatrixBufferHandle,
    grad_params: MatrixBufferHandle,
) {
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