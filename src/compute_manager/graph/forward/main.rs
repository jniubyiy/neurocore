// src/compute_manager/graph/forward/main.rs

use crate::compute_manager::dim_change::DynamicTensor;
use crate::compute_manager::graph::model::MixedModel;
use crate::compute_manager::graph::types::{DynamicBatchTensor, DynamicContext, Segment};

impl MixedModel {
    /// Прямой проход одного образца или батча (матрицы).
    pub fn forward(&self, input: DynamicTensor) -> (DynamicTensor, Vec<Vec<DynamicContext>>) {
        if input.batch_size() > 1 {
            let batch = match input {
                DynamicTensor::Dim1(t) => DynamicBatchTensor::Dim1(vec![t]),
                DynamicTensor::Dim2(t) => DynamicBatchTensor::Dim2(vec![t]),
                DynamicTensor::Dim3(t) => DynamicBatchTensor::Dim3(vec![t]),
                DynamicTensor::Dim4(t) => DynamicBatchTensor::Dim4(vec![t]),
            };
            let (out_batch, ctxs) = self.forward_batch(batch);
            let out = match out_batch {
                DynamicBatchTensor::Dim1(mut v) => DynamicTensor::Dim1(v.remove(0)),
                DynamicBatchTensor::Dim2(mut v) => DynamicTensor::Dim2(v.remove(0)),
                DynamicBatchTensor::Dim3(mut v) => DynamicTensor::Dim3(v.remove(0)),
                DynamicBatchTensor::Dim4(mut v) => DynamicTensor::Dim4(v.remove(0)),
            };
            return (out, ctxs);
        }

        let batch = match input {
            DynamicTensor::Dim1(t) => DynamicBatchTensor::Dim1(vec![t]),
            DynamicTensor::Dim2(t) => DynamicBatchTensor::Dim2(vec![t]),
            DynamicTensor::Dim3(t) => DynamicBatchTensor::Dim3(vec![t]),
            DynamicTensor::Dim4(t) => DynamicBatchTensor::Dim4(vec![t]),
        };
        let (out_batch, ctxs) = self.forward_batch(batch);
        let out = match out_batch {
            DynamicBatchTensor::Dim1(mut v) => DynamicTensor::Dim1(v.remove(0)),
            DynamicBatchTensor::Dim2(mut v) => DynamicTensor::Dim2(v.remove(0)),
            DynamicBatchTensor::Dim3(mut v) => DynamicTensor::Dim3(v.remove(0)),
            DynamicBatchTensor::Dim4(mut v) => DynamicTensor::Dim4(v.remove(0)),
        };
        (out, ctxs)
    }

    /// Прямой проход батча. Принимает DynamicBatchTensor, возвращает результат и контексты.
    pub fn forward_batch(
        &self,
        batch: DynamicBatchTensor,
    ) -> (DynamicBatchTensor, Vec<Vec<DynamicContext>>) {
        let params = self.store.lock().unwrap().all_params().to_vec();

        let mut streams: Vec<Vec<DynamicTensor>> = vec![match batch {
            DynamicBatchTensor::Dim1(tensors) => tensors
                .into_iter()
                .map(|t| DynamicTensor::Dim1(t))
                .collect(),
            DynamicBatchTensor::Dim2(tensors) => tensors
                .into_iter()
                .map(|t| DynamicTensor::Dim2(t))
                .collect(),
            DynamicBatchTensor::Dim3(tensors) => tensors
                .into_iter()
                .map(|t| DynamicTensor::Dim3(t))
                .collect(),
            DynamicBatchTensor::Dim4(tensors) => tensors
                .into_iter()
                .map(|t| DynamicTensor::Dim4(t))
                .collect(),
        }];

        let batch_size = streams[0].len();
        let mut all_ctxs: Vec<Vec<DynamicContext>> = vec![Vec::new(); batch_size];

        for seg in self.segments.iter() {
            match seg {
                Segment::Unsqueeze(target_dims) => {
                    self.process_unsqueeze_forward(&mut streams, target_dims);
                }
                Segment::ReduceMean(target_dims) => {
                    self.process_reduce_mean_forward(&mut streams, target_dims);
                }
                Segment::UniversalProcessor(proc, slices, stream_indices) => {
                    self.process_universal_processor_forward(
                        proc,
                        slices,
                        0,
                        &params,
                        &mut streams,
                        &mut all_ctxs,
                        stream_indices,
                    );
                }
                Segment::SplitterConnector { output_dims, .. } => {
                    self.process_splitter_connector_forward(
                        output_dims.clone(),
                        batch_size,
                        &mut streams,
                        &mut all_ctxs,
                    );
                }
                Segment::CombinerConnector { input_dims, .. } => {
                    self.process_combiner_connector_forward(
                        input_dims.clone(),
                        batch_size,
                        &mut streams,
                        &mut all_ctxs,
                    );
                }
            }
        }

        assert_eq!(streams.len(), 1, "Модель должна завершаться одним потоком данных");

        let out_batch = match streams[0].first().unwrap() {
            DynamicTensor::Dim1(_) => DynamicBatchTensor::Dim1(
                streams[0]
                    .iter()
                    .map(|d| match d {
                        DynamicTensor::Dim1(t) => t.clone(),
                        _ => unreachable!(),
                    })
                    .collect(),
            ),
            DynamicTensor::Dim2(_) => DynamicBatchTensor::Dim2(
                streams[0]
                    .iter()
                    .map(|d| match d {
                        DynamicTensor::Dim2(t) => t.clone(),
                        _ => unreachable!(),
                    })
                    .collect(),
            ),
            DynamicTensor::Dim3(_) => DynamicBatchTensor::Dim3(
                streams[0]
                    .iter()
                    .map(|d| match d {
                        DynamicTensor::Dim3(t) => t.clone(),
                        _ => unreachable!(),
                    })
                    .collect(),
            ),
            DynamicTensor::Dim4(_) => DynamicBatchTensor::Dim4(
                streams[0]
                    .iter()
                    .map(|d| match d {
                        DynamicTensor::Dim4(t) => t.clone(),
                        _ => unreachable!(),
                    })
                    .collect(),
            ),
        };

        (out_batch, all_ctxs)
    }
}