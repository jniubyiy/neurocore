// src/compute_manager/graph/forward/segments/processors.rs

use std::sync::Arc;

use crate::compute_manager::dim_change::DynamicTensor;
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
        _seg_idx: usize,
        params: &[f32],
        streams: &mut Vec<Vec<DynamicTensor>>,
        all_ctxs: &mut Vec<Vec<DynamicContext>>,
        stream_indices: &Option<Vec<usize>>,
    ) {
        // Определяем активные потоки (индексы входных потоков, которые будут обработаны)
        let active_indices: Vec<usize> = match stream_indices {
            Some(indices) => indices.clone(),
            None => (0..streams.len()).collect(),
        };

        // -------------------------------------------
        // GPU-путь: обработка всего батча одной командой
        // -------------------------------------------
        if let Some(ref gpu_compute_mutex) = self.gpu_compute {
            let gpu_compute = gpu_compute_mutex.lock().unwrap();
            let num_output_streams = streams.len();
            let mut new_streams: Vec<Option<Vec<DynamicTensor>>> = vec![None; num_output_streams];

            for &stream_idx in &active_indices {
                let batch_samples = &streams[stream_idx];
                // Преобразуем образцы в единую матрицу (batch, features)
                let input_mat = MixedModel::samples_to_mat(batch_samples);
                // Запускаем GPU-обработку всей цепочки слоёв
                let (out_mat, layer_ctxs) = process_forward_gpu(
                    &gpu_compute,
                    proc,
                    slices,
                    params,
                    &input_mat,
                );
                // Преобразуем выходную матрицу обратно в образцы
                let samples = MixedModel::mat_to_samples(&out_mat);
                new_streams[stream_idx] = Some(samples);

                // Сохраняем контексты для каждого образца
                for (_i, ctx_sample) in all_ctxs.iter_mut().enumerate() {
                    // На каждый образец добавляем полный набор контекстов слоёв
                    ctx_sample.extend(layer_ctxs.clone());
                }
            }

            // Заполняем потоки, которые не обрабатывались (остаются без изменений)
            for (i, opt) in new_streams.iter_mut().enumerate() {
                if opt.is_none() {
                    *opt = Some(streams[i].clone());
                }
            }
            *streams = new_streams.into_iter().map(|o| o.unwrap()).collect();
            return;
        }

        // -------------------------------------------
        // CPU-путь (исходная многопоточная реализация)
        // -------------------------------------------
        let num_output_streams = streams.len();
        let mut new_streams: Vec<Option<Vec<DynamicTensor>>> = vec![None; num_output_streams];

        let layers_arc = Arc::clone(proc);
        let slices_arc = Arc::new(slices.to_vec());

        let mut receivers = Vec::with_capacity(active_indices.len());

        for &stream_idx in &active_indices {
            let batch = streams[stream_idx].clone();
            let batch_len = batch.len();

            let assignment = self.executor.plan_chunks_assignment(batch_len);

            let layers = Arc::clone(&layers_arc);
            let slices = Arc::clone(&slices_arc);
            let params = params.to_vec();

            let (tx, rx) = std::sync::mpsc::channel();

            for (_worker_id, ranges) in assignment.iter().enumerate() {
                if ranges.is_empty() { continue; }
                let ranges = ranges.clone();
                let batch = batch.clone();
                let layers = Arc::clone(&layers);
                let slices = Arc::clone(&slices);
                let params = params.clone();
                let tx = tx.clone();
                let executor = self.executor.clone_executor();

                executor.execute_dyn(Box::new(move || {
                    let mut results: Vec<(usize, DynamicTensor)> = Vec::new();
                    let mut ctxs: Vec<(usize, Vec<DynamicContext>)> = Vec::new();

                    for (range_start, range_size) in &ranges {
                        let chunk_mat = MixedModel::samples_to_mat(
                            &batch[*range_start..*range_start + *range_size]
                        );
                        let (chunk_out_mat, chunk_ctxs) =
                            MixedModel::forward_universal_batch_mat(
                                &layers, &slices, &chunk_mat, &params,
                            );
                        let samples = MixedModel::mat_to_samples(&chunk_out_mat);
                        for (i, sample) in samples.into_iter().enumerate() {
                            results.push((*range_start + i, sample));
                        }
                        for i in 0..*range_size {
                            ctxs.push((*range_start + i, chunk_ctxs.clone()));
                        }
                    }
                    tx.send((results, ctxs)).ok();
                }));
            }
            receivers.push((stream_idx, rx));
        }

        self.executor.wait_all();

        for (stream_idx, rx) in receivers {
            let batch_len = streams[stream_idx].len();
            let mut stream_outputs: Vec<Option<DynamicTensor>> = vec![None; batch_len];
            let mut stream_ctxs: Vec<Vec<DynamicContext>> = vec![Vec::new(); batch_len];

            while let Ok((results, ctxs)) = rx.recv() {
                for (idx, out) in results { stream_outputs[idx] = Some(out); }
                for (idx, sample_ctxs) in ctxs { stream_ctxs[idx].extend(sample_ctxs); }
            }

            new_streams[stream_idx] = Some(stream_outputs.into_iter().map(|o| o.unwrap()).collect());
            for (_i, ctxs_sample) in stream_ctxs.into_iter().enumerate() {
                all_ctxs[_i].extend(ctxs_sample);
            }
        }

        for (i, opt) in new_streams.iter_mut().enumerate() {
            if opt.is_none() { *opt = Some(streams[i].clone()); }
        }
        *streams = new_streams.into_iter().map(|o| o.unwrap()).collect();
    }
}