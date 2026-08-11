// src/compute_manager/graph/forward/segments/processors.rs

use std::sync::Arc;
use std::time::Instant;

use faer::Mat;
use crate::compute_manager::cpu::worker_pool::WorkerPool;
use crate::compute_manager::graph::model::MixedModel;
use crate::compute_manager::graph::types::DynamicContext;
use crate::compute_manager::gpu::processor::process_forward_gpu;
use crate::layers::UniversalLayer;
use crate::model_plan::param_store::ParamSlice;

impl MixedModel {
    pub(crate) fn process_universal_processor_forward(
        &self,
        proc: &Arc<Vec<Box<dyn UniversalLayer>>>,
        slices: &[ParamSlice],
        _seg_index: usize,
        params: &[f32],
        stream_matrices: &mut Vec<Mat<f32>>,
        all_ctxs: &mut Vec<Vec<DynamicContext>>,
        stream_indices: &Option<Vec<usize>>,
    ) {
        // Определяем, какие входные потоки обрабатывать
        let active_indices: Vec<usize> = match stream_indices {
            Some(indices) => indices.clone(),
            None => (0..stream_matrices.len()).collect(),
        };

        // --- GPU‑путь ---
        if let Some(ref gpu_compute_mutex) = self.gpu_compute {
            eprintln!("[PROCESSOR] GPU path selected");
            let gpu_compute = gpu_compute_mutex.lock().unwrap();
            for &stream_idx in &active_indices {
                let input_mat = stream_matrices[stream_idx].clone();
                let (out_mat, layer_ctxs) = process_forward_gpu(
                    &gpu_compute,
                    proc,
                    slices,
                    params,
                    &input_mat,
                );
                stream_matrices[stream_idx] = out_mat;

                // Добавляем контексты для каждого сэмпла батча
                for sample_ctxs in all_ctxs.iter_mut() {
                    sample_ctxs.extend(layer_ctxs.clone());
                }
            }
            return;
        }

        eprintln!("[PROCESSOR] CPU path selected (no gpu_compute)");

        // --- CPU‑путь (многопоточный, работает с подматрицами) ---

        let layers_arc = Arc::clone(proc);
        let slices_arc = Arc::new(slices.to_vec());

        let mut receivers = Vec::with_capacity(active_indices.len());

        // Канал для сбора времени выполнения чанков на каждом CPU
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
                if ranges.is_empty() {
                    continue;
                }
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
                        let start = Instant::now();

                        // Извлекаем срез строк как независимую матрицу
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

                        let duration = start.elapsed().as_nanos() as f64;
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

        // Собираем все времена выполнения
        while let Ok((cpu_idx, chunk_size, duration_ns)) = time_rx.try_recv() {
            self.scheduler
                .lock()
                .unwrap()
                .report_execution_time(cpu_idx, chunk_size, duration_ns);
        }

        // Объединяем результаты чанков для каждого потока
        for (stream_idx, rx) in receivers {
            let batch_len = stream_matrices[stream_idx].nrows(); // исходное число строк (не меняется)

            // Временные хранилища для результатов и контекстов
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

            // Определяем новое количество признаков по первому чанку
            let new_features = chunk_results_list
                .first()
                .map(|(_, m)| m.ncols())
                .unwrap_or(0);

            // Создаём результирующую матрицу правильного размера
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

            // Обновляем матрицу потока
            stream_matrices[stream_idx] = result_matrix;

            // Добавляем контексты в общий вектор
            for (sample_idx, ctxs) in stream_ctxs.into_iter().enumerate() {
                all_ctxs[sample_idx].extend(ctxs);
            }
        }
    }
}