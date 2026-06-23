// src/compute_manager/graph/backward/main.rs

use crate::compute_manager::dim_change::DynamicTensor;
use crate::compute_manager::graph::model::MixedModel;
use crate::compute_manager::graph::types::{DynamicBatchTensor, DynamicContext, Segment};

impl MixedModel {
    /// Обратный проход одного образца или батча (матрицы градиентов).
    pub fn backward(
        &self,
        contexts: &[Vec<DynamicContext>],
        delta: DynamicTensor,
    ) -> (DynamicTensor, Vec<Vec<f32>>) {
        // Если батч > 1, обрабатываем как батч
        if delta.batch_size() > 1 {
            let batch = match delta {
                DynamicTensor::Dim1(t) => DynamicBatchTensor::Dim1(vec![t]),
                DynamicTensor::Dim2(t) => DynamicBatchTensor::Dim2(vec![t]),
                DynamicTensor::Dim3(t) => DynamicBatchTensor::Dim3(vec![t]),
                DynamicTensor::Dim4(t) => DynamicBatchTensor::Dim4(vec![t]),
            };
            let (in_batch, grad) = self.backward_batch(contexts, batch);
            let in_tensor = match in_batch {
                DynamicBatchTensor::Dim1(mut v) => DynamicTensor::Dim1(v.remove(0)),
                DynamicBatchTensor::Dim2(mut v) => DynamicTensor::Dim2(v.remove(0)),
                DynamicBatchTensor::Dim3(mut v) => DynamicTensor::Dim3(v.remove(0)),
                DynamicBatchTensor::Dim4(mut v) => DynamicTensor::Dim4(v.remove(0)),
            };
            return (in_tensor, vec![grad]);
        }

        // Одиночный образец (batch = 1) – заворачиваем в батч из одного элемента
        let batch = match delta {
            DynamicTensor::Dim1(t) => DynamicBatchTensor::Dim1(vec![t]),
            DynamicTensor::Dim2(t) => DynamicBatchTensor::Dim2(vec![t]),
            DynamicTensor::Dim3(t) => DynamicBatchTensor::Dim3(vec![t]),
            DynamicTensor::Dim4(t) => DynamicBatchTensor::Dim4(vec![t]),
        };
        let (in_batch, grad) = self.backward_batch(contexts, batch);
        let in_tensor = match in_batch {
            DynamicBatchTensor::Dim1(mut v) => DynamicTensor::Dim1(v.remove(0)),
            DynamicBatchTensor::Dim2(mut v) => DynamicTensor::Dim2(v.remove(0)),
            DynamicBatchTensor::Dim3(mut v) => DynamicTensor::Dim3(v.remove(0)),
            DynamicBatchTensor::Dim4(mut v) => DynamicTensor::Dim4(v.remove(0)),
        };
        (in_tensor, vec![grad])
    }

    /// Обратный проход батча.
    pub fn backward_batch(
        &self,
        contexts: &[Vec<DynamicContext>],
        delta: DynamicBatchTensor,
    ) -> (DynamicBatchTensor, Vec<f32>) {
        let batch_size = match &delta {
            DynamicBatchTensor::Dim1(v) => v.len(),
            DynamicBatchTensor::Dim2(v) => v.len(),
            DynamicBatchTensor::Dim3(v) => v.len(),
            DynamicBatchTensor::Dim4(v) => v.len(),
        };
        let params = self.store.lock().unwrap().all_params().to_vec();
        let param_len = params.len();
        let mut total_grad = vec![0.0f32; param_len];

        // Преобразуем батч градиентов в потоки (каждый элемент – отдельный сэмпл)
        let mut streams: Vec<Vec<DynamicTensor>> = vec![match delta {
            DynamicBatchTensor::Dim1(v) => v.into_iter().map(DynamicTensor::Dim1).collect(),
            DynamicBatchTensor::Dim2(v) => v.into_iter().map(DynamicTensor::Dim2).collect(),
            DynamicBatchTensor::Dim3(v) => v.into_iter().map(DynamicTensor::Dim3).collect(),
            DynamicBatchTensor::Dim4(v) => v.into_iter().map(DynamicTensor::Dim4).collect(),
        }];

        let total_context_len = contexts[0].len();
        let mut ctx_pos = total_context_len;

        for seg in self.segments.iter().rev() {
            match seg {
                Segment::Unsqueeze(target_dims) => {
                    self.process_unsqueeze_backward(&mut streams, target_dims);
                }
                Segment::ReduceMean(target_dims) => {
                    self.process_reduce_mean_backward(&mut streams, target_dims);
                }
                Segment::UniversalProcessor(proc, slices, stream_indices) => {
                    let result = self.process_universal_processor_backward(
                        proc,
                        slices,
                        &streams,
                        contexts,
                        ctx_pos,
                        batch_size,
                        &params,
                        &mut total_grad,
                        stream_indices,
                    );
                    streams = result.0;
                    ctx_pos = result.1;
                }
                Segment::SplitterConnector { output_dims, .. } => {
                    let result = self.process_splitter_connector_backward(
                        &streams, output_dims.clone(), batch_size, ctx_pos,
                    );
                    streams = result.0;
                    ctx_pos = result.1;
                }
                Segment::CombinerConnector { input_dims, .. } => {
                    let result = self.process_combiner_connector_backward(
                        &streams, input_dims.clone(), batch_size, ctx_pos,
                    );
                    streams = result.0;
                    ctx_pos = result.1;
                }
            }
        }

        assert_eq!(streams.len(), 1, "После backward должен остаться один входной поток градиентов");

        let out_batch = match streams[0].first().unwrap() {
            DynamicTensor::Dim1(_) => DynamicBatchTensor::Dim1(
                streams[0].iter().map(|d| match d {
                    DynamicTensor::Dim1(t) => t.clone(),
                    _ => unreachable!(),
                }).collect(),
            ),
            DynamicTensor::Dim2(_) => DynamicBatchTensor::Dim2(
                streams[0].iter().map(|d| match d {
                    DynamicTensor::Dim2(t) => t.clone(),
                    _ => unreachable!(),
                }).collect(),
            ),
            DynamicTensor::Dim3(_) => DynamicBatchTensor::Dim3(
                streams[0].iter().map(|d| match d {
                    DynamicTensor::Dim3(t) => t.clone(),
                    _ => unreachable!(),
                }).collect(),
            ),
            DynamicTensor::Dim4(_) => DynamicBatchTensor::Dim4(
                streams[0].iter().map(|d| match d {
                    DynamicTensor::Dim4(t) => t.clone(),
                    _ => unreachable!(),
                }).collect(),
            ),
        };

        (out_batch, total_grad)
    }
}