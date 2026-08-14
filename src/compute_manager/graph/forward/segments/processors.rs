// src/compute_manager/graph/forward/segments/processors.rs

use std::sync::Arc;
use std::time::Instant;

use faer::Mat;
use crate::compute_manager::cpu::worker_pool::WorkerPool;
use crate::compute_manager::graph::model::MixedModel;
use crate::compute_manager::graph::types::DynamicContext;
use crate::compute_manager::gpu::processor::process_forward_gpu;
use crate::compute_manager::matrix_buffer::{MatrixBufferHandle, TempMatrixPool};
use crate::compute_manager::persistent_buffer::SegmentPersistentBuffers;
use crate::layers::{UniversalLayer, UniversalLayerBuffered};
use crate::layers::buffered_context::BufferedContext;
use crate::layers::mat_context::MatContext;
use crate::model_plan::param_store::ParamSlice;

impl MixedModel {
    pub(crate) fn process_universal_processor_forward(
        &mut self,
        proc: &Arc<Vec<Box<dyn UniversalLayer>>>,
        slices: &[ParamSlice],
        seg_index: usize,
        params: &[f32],
        stream_matrices: &mut Vec<Mat<f32>>,
        all_ctxs: &mut Vec<Vec<DynamicContext>>,
        stream_indices: &Option<Vec<usize>>,
    ) {
        let active_indices: Vec<usize> = match stream_indices {
            Some(indices) => indices.clone(),
            None => (0..stream_matrices.len()).collect(),
        };

        if let Some(ref gpu_compute_mutex) = self.gpu_compute {
            let buffers = self.get_segment_buffers(seg_index);
            let gpu_compute = gpu_compute_mutex.lock().unwrap();

            for &stream_idx in &active_indices {
                let input_mat = stream_matrices[stream_idx].clone();
                let (out_mat, layer_ctxs) = if let Some(ref buffers) = buffers {
                    process_forward_gpu(
                        &gpu_compute,
                        buffers,
                        proc,
                        slices,
                        params,
                        &input_mat,
                    )
                } else {
                    let temp_buffers = SegmentPersistentBuffers::for_segment(
                        &self.segments[seg_index],
                        &self.segment_placement[seg_index].compute_device,
                        input_mat.nrows(),
                        &mut self.memory_executor.lock().unwrap(),
                    );
                    let result = process_forward_gpu(
                        &gpu_compute,
                        &temp_buffers,
                        proc,
                        slices,
                        params,
                        &input_mat,
                    );
                    result
                };
                stream_matrices[stream_idx] = out_mat;
                for sample_ctxs in all_ctxs.iter_mut() {
                    sample_ctxs.extend(layer_ctxs.clone());
                }
            }
            return;
        }

        let layers_arc = Arc::clone(proc);
        let slices_arc = Arc::new(slices.to_vec());
        let mut receivers = Vec::with_capacity(active_indices.len());
        let (time_tx, time_rx) = std::sync::mpsc::channel();

        for &stream_idx in &active_indices {
            let full_matrix = stream_matrices[stream_idx].clone();
            let batch_len = full_matrix.nrows();
            let assignment = self.executor.plan_chunks_assignment(batch_len);

            let layers = Arc::clone(&layers_arc);
            let slices = Arc::clone(&slices_arc);
            let params = params.to_vec();

            let (tx, rx) = std::sync::mpsc::channel();
            let time_tx = time_tx.clone();

            for (_worker_id, ranges) in assignment.iter().enumerate() {
                if ranges.is_empty() { continue; }
                let ranges = ranges.clone();
                let matrix_chunk = full_matrix.clone();
                let layers = Arc::clone(&layers);
                let slices = Arc::clone(&slices);
                let params = params.clone();
                let tx = tx.clone();
                let time_tx = time_tx.clone();
                let executor = self.executor.clone_executor();

                executor.execute_dyn(Box::new(move || {
                    let cpu_idx = WorkerPool::current_worker_index();
                    let mut results: Vec<(usize, Mat<f32>)> = Vec::new();
                    let mut ctxs: Vec<(usize, Vec<DynamicContext>)> = Vec::new();

                    for (range_start, range_size) in &ranges {
                        let chunk_start = Instant::now();
                        let chunk_mat = matrix_chunk
                            .submatrix(*range_start, 0, *range_size, matrix_chunk.ncols())
                            .to_owned();
                        let (chunk_out_mat, chunk_ctxs) =
                            MixedModel::forward_universal_batch_mat(
                                &layers,
                                &slices,
                                &chunk_mat,
                                &params,
                            );
                        let duration = chunk_start.elapsed().as_nanos() as f64;
                        let _ = time_tx.send((cpu_idx, *range_size, duration));
                        results.push((*range_start, chunk_out_mat));
                        for i in 0..*range_size {
                            ctxs.push((*range_start + i, chunk_ctxs.clone()));
                        }
                    }
                    let _ = tx.send((results, ctxs));
                }));
            }
            receivers.push((stream_idx, rx));
        }

        self.executor.wait_all();
        while let Ok((cpu_idx, chunk_size, duration_ns)) = time_rx.try_recv() {
            self.scheduler
                .lock()
                .unwrap()
                .report_execution_time(cpu_idx, chunk_size, duration_ns);
        }

        for (stream_idx, rx) in receivers {
            let batch_len = stream_matrices[stream_idx].nrows();
            let mut chunk_results_list: Vec<(usize, Mat<f32>)> = Vec::new();
            let mut stream_ctxs: Vec<Vec<DynamicContext>> = vec![Vec::new(); batch_len];

            while let Ok((chunk_results, chunk_ctxs)) = rx.recv() {
                for (row_offset, chunk_mat) in chunk_results {
                    chunk_results_list.push((row_offset, chunk_mat));
                }
                for (sample_idx, sample_ctxs) in chunk_ctxs {
                    stream_ctxs[sample_idx].extend(sample_ctxs);
                }
            }

            let new_features = chunk_results_list.first().map(|(_, m)| m.ncols()).unwrap_or(0);
            let mut result_matrix = Mat::zeros(batch_len, new_features);
            for (row_offset, chunk_mat) in chunk_results_list {
                let rows = chunk_mat.nrows();
                let cols = chunk_mat.ncols();
                assert_eq!(cols, new_features, "Chunk column count mismatch");
                for r in 0..rows {
                    for c in 0..cols {
                        result_matrix[(row_offset + r, c)] = chunk_mat[(r, c)];
                    }
                }
            }
            stream_matrices[stream_idx] = result_matrix;
            for (sample_idx, ctxs) in stream_ctxs.into_iter().enumerate() {
                all_ctxs[sample_idx].extend(ctxs);
            }
        }
    }

    // CPU-путь с использованием MatrixBufferHandle
    pub(crate) fn process_universal_processor_forward_buffered(
        &mut self,
        pool: &mut TempMatrixPool,
        proc: &Arc<Vec<Box<dyn UniversalLayer>>>,
        slices: &[ParamSlice],
        _seg_index: usize,
        params: &[f32],
        stream_buffers: &mut Vec<MatrixBufferHandle>,
        all_ctxs: &mut Vec<Vec<DynamicContext>>,
        stream_indices: &Option<Vec<usize>>,
    ) {
        let active_indices: Vec<usize> = match stream_indices {
            Some(indices) => indices.clone(),
            None => (0..stream_buffers.len()).collect(),
        };

        let layers = proc.as_ref();
        let num_layers = layers.len();

        // Клонируем входные дескрипторы для неактивных потоков (они не меняются)
        let mut new_stream: Vec<Option<MatrixBufferHandle>> = stream_buffers
            .iter()
            .map(|handle| Some(handle.clone()))
            .collect();

        for &stream_idx in &active_indices {
            // Забираем входной дескриптор (клонируем, исходный останется для других)
            let input_handle = stream_buffers[stream_idx].clone();
            let batch_size = input_handle.rows();
            let mut current_input = input_handle;
            let mut layer_ctxs: Vec<DynamicContext> = Vec::with_capacity(num_layers);

            for i in 0..num_layers {
                let layer = &layers[i];
                let slice = &slices[i];

                // Определяем размер выходного буфера
                let out_features = get_buffered_output_features(layer, &current_input);
                let output_handle = pool.acquire(batch_size, out_features);

                // Выполняем прямой проход
                call_forward_buffered(
                    layer,
                    &current_input,
                    &output_handle,
                    params,
                    slice,
                );

                // Создаём контекст для обратного прохода
                let buffered_ctx = build_buffered_context(layer, &current_input, &output_handle);
                layer_ctxs.push(DynamicContext::Buffered(buffered_ctx));

                // Обновляем текущий вход для следующего слоя
                current_input = output_handle;
            }

            // Записываем результат для этого потока
            new_stream[stream_idx] = Some(current_input);

            // Добавляем контексты для всех сэмплов (одинаковы для всех)
            for sample_ctxs in all_ctxs.iter_mut() {
                sample_ctxs.extend(layer_ctxs.clone());
            }
        }

        // Обновляем stream_buffers
        *stream_buffers = new_stream
            .into_iter()
            .map(|opt| opt.expect("Missing stream buffer after forward"))
            .collect();
    }
}

/// Возвращает количество выходных признаков слоя, используя UniversalLayerBuffered.
fn get_buffered_output_features(layer: &Box<dyn UniversalLayer>, input: &MatrixBufferHandle) -> usize {
    if let Some(l) = layer.as_linear() {
        <dyn UniversalLayerBuffered>::output_features(l)
    } else if layer.as_relu().is_some()
        || layer.as_sigmoid().is_some()
        || layer.as_tanh().is_some()
        || layer.as_leaky_relu().is_some()
        || layer.as_identity().is_some()
        || layer.as_softmax().is_some()
        || layer.as_memory().is_some()
        || layer.as_soft_sparse_gate().is_some()
        || layer.as_soft_keep_gate().is_some()
        || layer.as_dual_anchor().is_some()
    {
        // Для этих слоёв выходная размерность равна входной
        input.cols()
    } else {
        // Fallback
        input.cols()
    }
}

/// Вызывает буферизованный прямой проход для слоя.
fn call_forward_buffered(
    layer: &Box<dyn UniversalLayer>,
    input: &MatrixBufferHandle,
    output: &MatrixBufferHandle,
    params: &[f32],
    slice: &ParamSlice,
) {
    if let Some(l) = layer.as_linear() {
        <dyn UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice)
    } else if let Some(l) = layer.as_relu() {
        <dyn UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice)
    } else if let Some(l) = layer.as_sigmoid() {
        <dyn UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice)
    } else if let Some(l) = layer.as_tanh() {
        <dyn UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice)
    } else if let Some(l) = layer.as_leaky_relu() {
        <dyn UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice)
    } else if let Some(l) = layer.as_identity() {
        <dyn UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice)
    } else if let Some(l) = layer.as_softmax() {
        <dyn UniversalLayerBuffered>::forward_buffered(l, input, output, params, slice)
    } else {
        // Fallback на старый метод (если слой не переведён)
        let input_mat = input.read();
        let input_slice = input_mat.as_slice().expect("CPU buffer");
        let mat_in = Mat::from_fn(input.rows(), input.cols(), |r, c| {
            input_slice[c * input.rows() + r]
        });
        let (mat_out, _ctx) = layer.forward_mat(&mat_in, params, slice);
        let mut output_guard = output.write();
        let output_slice = output_guard.as_slice_mut().expect("CPU buffer");
        for c in 0..mat_out.ncols() {
            for r in 0..mat_out.nrows() {
                output_slice[c * output.rows() + r] = mat_out[(r, c)];
            }
        }
    }
}

/// Создаёт буферизованный контекст для слоя.
///
/// В зависимости от типа слоя возвращает соответствующий вариант `BufferedContext`,
/// сохраняя `MatrixBufferHandle` на входные или выходные данные.
fn build_buffered_context(
    layer: &Box<dyn UniversalLayer>,
    input: &MatrixBufferHandle,
    output: &MatrixBufferHandle,
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
    } else {
        // Fallback: Identity
        BufferedContext::Identity { input: input.clone() }
    }
}