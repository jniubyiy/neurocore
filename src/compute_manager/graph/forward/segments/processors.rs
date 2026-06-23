// src/compute_manager/graph/forward/segments/processors.rs

use std::sync::Arc;

use crate::compute_manager::dim_change::DynamicTensor;
use crate::compute_manager::graph::model::MixedModel;
use crate::compute_manager::graph::types::DynamicContext;
use crate::layers::UniversalLayer;
use crate::model_plan::param_store::ParamSlice;

impl MixedModel {
    /// Единый прямой проход для группы универсальных слоёв (матричная оптимизация).
    /// Каждый чанк собирается в один DynamicTensor (батч), после чего последовательно
    /// прогоняется через все слои процессора, используя их матричную реализацию `forward`.
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
        let active_indices: Vec<usize> = match stream_indices {
            Some(indices) => indices.clone(),
            None => (0..streams.len()).collect(),
        };

        let num_output_streams = streams.len();
        let mut new_streams: Vec<Option<Vec<DynamicTensor>>> = vec![None; num_output_streams];

        let layers_arc = Arc::clone(proc);
        let slices_arc = Arc::new(slices.to_vec());

        let mut receivers = Vec::with_capacity(active_indices.len());

        for &stream_idx in &active_indices {
            let batch = streams[stream_idx].clone(); // Vec<DynamicTensor> отдельных сэмплов
            let batch_len = batch.len();

            // Получаем распределение диапазонов по ядрам
            let assignment = self.scheduler.lock().unwrap().plan_chunks_assignment(batch_len);

            let layers = Arc::clone(&layers_arc);
            let slices = Arc::clone(&slices_arc);
            let params = params.to_vec();
            let pool = self.pool.clone();

            let (tx, rx) = std::sync::mpsc::channel();

            for (_worker_id, ranges) in assignment.iter().enumerate() {
                if ranges.is_empty() {
                    continue;
                }
                let ranges = ranges.clone();
                let batch = batch.clone();
                let layers = Arc::clone(&layers);
                let slices = Arc::clone(&slices);
                let params = params.clone();
                let tx = tx.clone();

                pool.execute(move || {
                    let mut results: Vec<(usize, DynamicTensor)> = Vec::new();
                    let mut ctxs: Vec<(usize, Vec<DynamicContext>)> = Vec::new();

                    for (range_start, range_size) in &ranges {
                        // Собираем сэмплы этого чанка в один DynamicTensor (батч)
                        let chunk_batch = MixedModel::samples_to_batch(&batch[*range_start..*range_start + *range_size]);

                        // Пропускаем через все слои процессора
                        let (chunk_out, chunk_ctxs) =
                            MixedModel::forward_universal_batch(&layers, &slices, &chunk_batch, &params);

                        // Разбиваем результат обратно на отдельные сэмплы
                        let samples = MixedModel::batch_to_samples(&chunk_out);
                        for (i, sample) in samples.into_iter().enumerate() {
                            results.push((*range_start + i, sample));
                        }
                        // Контексты для каждого сэмпла в чанке (пока все одинаковые)
                        for i in 0..*range_size {
                            ctxs.push((*range_start + i, chunk_ctxs.clone()));
                        }
                    }
                    tx.send((results, ctxs)).ok();
                });
            }
            receivers.push((stream_idx, rx));
        }

        self.pool.wait_all();

        // Сбор результатов
        for (stream_idx, rx) in receivers {
            let batch_len = streams[stream_idx].len();
            let mut stream_outputs: Vec<Option<DynamicTensor>> = vec![None; batch_len];
            let mut stream_ctxs: Vec<Vec<DynamicContext>> = vec![Vec::new(); batch_len];

            while let Ok((results, ctxs)) = rx.recv() {
                for (idx, out) in results {
                    stream_outputs[idx] = Some(out);
                }
                for (idx, sample_ctxs) in ctxs {
                    stream_ctxs[idx].extend(sample_ctxs);
                }
            }

            new_streams[stream_idx] = Some(stream_outputs.into_iter().map(|o| o.unwrap()).collect());
            for (i, ctxs_sample) in stream_ctxs.into_iter().enumerate() {
                all_ctxs[i].extend(ctxs_sample);
            }
        }

        // Заполняем неизменённые потоки
        for (i, opt) in new_streams.iter_mut().enumerate() {
            if opt.is_none() {
                *opt = Some(streams[i].clone());
            }
        }
        *streams = new_streams.into_iter().map(|o| o.unwrap()).collect();
    }

    /// Преобразует срез отдельных сэмплов в один DynamicTensor (батч).
    fn samples_to_batch(samples: &[DynamicTensor]) -> DynamicTensor {
        if samples.is_empty() {
            panic!("samples_to_batch: empty slice");
        }
        match &samples[0] {
            DynamicTensor::Dim1(_) => {
                let mut data = Vec::with_capacity(samples.len());
                for s in samples {
                    if let DynamicTensor::Dim1(t) = s {
                        data.push(t.data[0].clone()); // каждый сэмпл - это одна строка (Tensor2D с batch=1)
                    } else {
                        panic!("Mixed dimensions in batch");
                    }
                }
                DynamicTensor::Dim1(crate::tensor::Tensor2D::new(data))
            }
            DynamicTensor::Dim2(_) => {
                let mut data = Vec::with_capacity(samples.len());
                for s in samples {
                    if let DynamicTensor::Dim2(t) = s {
                        data.push(t.data[0].clone());
                    } else {
                        panic!("Mixed dimensions");
                    }
                }
                DynamicTensor::Dim2(crate::tensor::Tensor3D::new(data))
            }
            DynamicTensor::Dim3(_) => {
                let mut data = Vec::with_capacity(samples.len());
                for s in samples {
                    if let DynamicTensor::Dim3(t) = s {
                        data.push(t.data[0].clone());
                    }
                }
                DynamicTensor::Dim3(crate::tensor::Tensor4D::new(data))
            }
            DynamicTensor::Dim4(_) => {
                let mut data = Vec::with_capacity(samples.len());
                for s in samples {
                    if let DynamicTensor::Dim4(t) = s {
                        data.push(t.data[0].clone());
                    }
                }
                DynamicTensor::Dim4(crate::tensor::Tensor5D::new(data))
            }
        }
    }

    /// Разбивает батч (DynamicTensor) на отдельные сэмплы (каждый с batch=1).
    fn batch_to_samples(batch: &DynamicTensor) -> Vec<DynamicTensor> {
        match batch {
            DynamicTensor::Dim1(t) => {
                let mut samples = Vec::with_capacity(t.dim1);
                for r in 0..t.dim1 {
                    samples.push(DynamicTensor::Dim1(crate::tensor::Tensor2D::new(vec![t.data[r].clone()])));
                }
                samples
            }
            DynamicTensor::Dim2(t) => {
                let mut samples = Vec::with_capacity(t.dim1);
                for i in 0..t.dim1 {
                    samples.push(DynamicTensor::Dim2(crate::tensor::Tensor3D::new(vec![t.data[i].clone()])));
                }
                samples
            }
            DynamicTensor::Dim3(t) => {
                let mut samples = Vec::with_capacity(t.dim1);
                for i in 0..t.dim1 {
                    samples.push(DynamicTensor::Dim3(crate::tensor::Tensor4D::new(vec![t.data[i].clone()])));
                }
                samples
            }
            DynamicTensor::Dim4(t) => {
                let mut samples = Vec::with_capacity(t.dim1);
                for i in 0..t.dim1 {
                    samples.push(DynamicTensor::Dim4(crate::tensor::Tensor5D::new(vec![t.data[i].clone()])));
                }
                samples
            }
        }
    }

    /// Применяет все слои процессора к одному батчу (DynamicTensor) и возвращает результат и контексты.
    fn forward_universal_batch(
        layers: &[Box<dyn UniversalLayer>],
        slices: &[ParamSlice],
        batch: &DynamicTensor,
        params: &[f32],
    ) -> (DynamicTensor, Vec<DynamicContext>) {
        let mut current = batch.clone();
        let mut ctxs = Vec::new();
        for (layer, slice) in layers.iter().zip(slices.iter()) {
            let (next, ctx) = layer.forward(&current, params, slice);
            ctxs.push(ctx);
            current = next;
        }
        (current, ctxs)
    }
}