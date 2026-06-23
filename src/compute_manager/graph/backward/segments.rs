// src/compute_manager/graph/backward/segments.rs

use std::sync::{Arc, mpsc};

use crate::tensor::Tensor2D;
use crate::model_plan::param_store::ParamSlice;
use crate::compute_manager::dim_change::{self, DynamicTensor};
use crate::compute_manager::graph::model::MixedModel;
use crate::compute_manager::graph::types::DynamicContext;
use crate::layers::UniversalLayer;
use crate::layers::splitter_connector::SplitterConnector;
use crate::layers::combiner_connector::CombinerConnector;

impl MixedModel {
    // ---------- Операции изменения размерности ----------
    pub(super) fn process_unsqueeze_backward(
        &self,
        streams: &mut Vec<Vec<DynamicTensor>>,
        target_dims: &[usize],
    ) {
        let target_dims = target_dims.to_vec();
        for stream in streams.iter_mut() {
            for d in stream.iter_mut() {
                *d = dim_change::reduce_to(d.clone(), target_dims.clone());
            }
        }
    }

    pub(super) fn process_reduce_mean_backward(
        &self,
        streams: &mut Vec<Vec<DynamicTensor>>,
        target_dims: &[usize],
    ) {
        let target_dims = target_dims.to_vec();
        for stream in streams.iter_mut() {
            for d in stream.iter_mut() {
                *d = dim_change::unsqueeze_to(d.clone(), target_dims.clone());
            }
        }
    }

    // ---------- Универсальный процессор (обратный проход) ----------
    pub(super) fn process_universal_processor_backward(
        &self,
        proc: &Arc<Vec<Box<dyn UniversalLayer>>>,
        slices: &[ParamSlice],
        streams: &Vec<Vec<DynamicTensor>>,
        contexts: &[Vec<DynamicContext>],
        ctx_pos: usize,
        batch_size: usize,
        params: &[f32],
        total_grad: &mut Vec<f32>,
        stream_indices: &Option<Vec<usize>>,
    ) -> (Vec<Vec<DynamicTensor>>, usize) {
        let num_layers = proc.len();
        let num_input_streams = streams.len();

        let active_indices: Vec<usize> = match stream_indices {
            Some(indices) => indices.clone(),
            None => (0..num_input_streams).collect(),
        };

        let mut new_streams: Vec<Option<Vec<DynamicTensor>>> = vec![None; num_input_streams];

        let assignment = self.scheduler.lock().unwrap().plan_chunks_assignment(batch_size);

        if active_indices.len() == 1 {
            let stream_idx = active_indices[0];
            let stream_start = ctx_pos - num_layers;
            let deltas = streams[stream_idx].clone();

            let layers = Arc::clone(proc);
            let slices = slices.to_vec();
            let params = params.to_vec();

            let (tx, rx) = mpsc::channel();
            let mut tasks_sent = 0;

            for (_worker_id, ranges) in assignment.iter().enumerate() {
                if ranges.is_empty() {
                    continue;
                }
                let ranges = ranges.clone();
                let deltas = deltas.clone();
                let contexts = contexts.to_vec();
                let layers = Arc::clone(&layers);
                let slices = slices.clone();
                let params = params.clone();
                let tx = tx.clone();
                let pool = self.pool.clone();

                pool.execute(move || {
                    let mut local_grad = vec![0.0f32; params.len()];
                    let mut chunk_in_deltas: Vec<(usize, DynamicTensor)> = Vec::new();

                    for (range_start, range_size) in &ranges {
                        for i in 0..*range_size {
                            let sample_idx = *range_start + i;
                            let sample_ctxs: Vec<&DynamicContext> = 
                                contexts[sample_idx][stream_start..stream_start + num_layers]
                                    .iter()
                                    .collect();
                            let (in_delta, grad) = MixedModel::universal_backward_one_static(
                                &layers,
                                &slices,
                                &sample_ctxs,
                                &deltas[sample_idx],
                                &params,
                            );
                            chunk_in_deltas.push((sample_idx, in_delta));
                            for (idx, &g) in grad.iter().enumerate() {
                                local_grad[idx] += g;
                            }
                        }
                    }
                    tx.send((chunk_in_deltas, local_grad)).ok();
                });
                tasks_sent += 1;
            }

            self.pool.wait_all();

            let mut new_stream = vec![DynamicTensor::Dim1(Tensor2D::zeros(1, 1)); batch_size];
            for _ in 0..tasks_sent {
                if let Ok((chunk_in_deltas, local_grad)) = rx.recv() {
                    for (sample_idx, in_delta) in chunk_in_deltas {
                        new_stream[sample_idx] = in_delta;
                    }
                    for (idx, &g) in local_grad.iter().enumerate() {
                        total_grad[idx] += g;
                    }
                }
            }

            new_streams[stream_idx] = Some(new_stream);
        } else {
            let mut sorted_active: Vec<usize> = active_indices.iter().cloned().collect();
            sorted_active.sort_unstable();
            let mut receivers = Vec::with_capacity(active_indices.len());

            for &stream_idx in &active_indices {
                let pos_in_sorted = sorted_active.iter().position(|&x| x == stream_idx).unwrap();
                let stream_start = ctx_pos - (sorted_active.len() - pos_in_sorted) * num_layers;

                let sample_contexts: Vec<Vec<DynamicContext>> = (0..batch_size)
                    .map(|i| {
                        contexts[i][stream_start..stream_start + num_layers]
                            .iter()
                            .cloned()
                            .collect()
                    })
                    .collect();

                let deltas: Vec<DynamicTensor> = streams[stream_idx].clone();

                let proc_arc = proc.clone();
                let slices = slices.to_vec();
                let params = params.to_vec();
                let (tx, rx) = mpsc::channel();
                receivers.push((stream_idx, rx));

                let pool = self.pool.clone();
                pool.execute(move || {
                    let mut local_grad = vec![0.0f32; params.len()];
                    let mut new_stream = Vec::with_capacity(batch_size);
                    for i in 0..batch_size {
                        let delta_t = deltas[i].clone();
                        let sample_ctxs: Vec<&DynamicContext> = sample_contexts[i].iter().collect();
                        let (in_delta, grad) = MixedModel::universal_backward_one_static(
                            &proc_arc,
                            &slices,
                            &sample_ctxs,
                            &delta_t,
                            &params,
                        );
                        new_stream.push(in_delta);
                        for (idx, &g) in grad.iter().enumerate() {
                            local_grad[idx] += g;
                        }
                    }
                    tx.send((new_stream, local_grad)).ok();
                });
            }

            self.pool.wait_all();

            for (stream_idx, rx) in receivers {
                if let Ok((stream_data, local_grad)) = rx.recv() {
                    new_streams[stream_idx] = Some(stream_data);
                    for (idx, &g) in local_grad.iter().enumerate() {
                        total_grad[idx] += g;
                    }
                }
            }
        }

        for (i, opt) in new_streams.iter_mut().enumerate() {
            if opt.is_none() {
                *opt = Some(streams[i].clone());
            }
        }

        let new_ctx_pos = ctx_pos - num_layers * active_indices.len();
        (new_streams.into_iter().map(|o| o.unwrap()).collect(), new_ctx_pos)
    }

    pub(crate) fn universal_backward_one_static(
        layers: &[Box<dyn UniversalLayer>],
        slices: &[ParamSlice],
        ctxs: &[&DynamicContext],
        delta: &DynamicTensor,
        params: &[f32],
    ) -> (DynamicTensor, Vec<f32>) {
        let num_layers = layers.len();
        assert_eq!(ctxs.len(), num_layers);
        let mut current_delta = delta.clone();
        let mut total_grad = vec![0.0f32; params.len()];

        for i in (0..num_layers).rev() {
            let (in_delta, grad) = layers[i].backward(ctxs[i], &current_delta, params, &slices[i]);
            current_delta = in_delta;
            for (idx, &g) in grad.iter().enumerate() {
                total_grad[idx] += g;
            }
        }
        (current_delta, total_grad)
    }

    // ---------- Коннекторы ----------

    pub(super) fn process_splitter_connector_backward(
        &self,
        streams: &Vec<Vec<DynamicTensor>>,
        output_dims: Vec<usize>,
        batch_size: usize,
        ctx_pos: usize,
    ) -> (Vec<Vec<DynamicTensor>>, usize) {
        assert_eq!(
            streams.len(),
            output_dims.len(),
            "SplitterConnector backward: ожидается {} потоков градиентов",
            output_dims.len()
        );

        let num_outputs = output_dims.len();
        let input_dim: usize = output_dims.iter().sum();

        let mut deltas = Vec::with_capacity(num_outputs);
        for out_idx in 0..num_outputs {
            let stream = &streams[out_idx];
            let mut data = Vec::with_capacity(batch_size);
            for d in stream.iter() {
                if let DynamicTensor::Dim1(t) = d {
                    data.push(t.data[0].clone());
                } else {
                    panic!("SplitterConnector backward: ожидается Dim1");
                }
            }
            deltas.push(DynamicTensor::Dim1(Tensor2D::new(data)));
        }

        let ctx = DynamicContext::Ctx1D(
            crate::layers::context1d::LayerContext1D::SplitterConnector {
                input: Tensor2D::zeros(1, input_dim),
            },
        );

        let connector = SplitterConnector::new(input_dim, output_dims.clone());
        let (in_delta, _) = connector.backward(&ctx, &deltas);

        let mut combined_stream = Vec::with_capacity(batch_size);
        if let DynamicTensor::Dim1(t) = &in_delta {
            for r in 0..batch_size {
                combined_stream.push(DynamicTensor::Dim1(Tensor2D::new(vec![t.data[r].clone()])));
            }
        } else {
            panic!("SplitterConnector backward вернул не Dim1");
        }

        (vec![combined_stream], ctx_pos - 1)
    }

    pub(super) fn process_combiner_connector_backward(
        &self,
        streams: &Vec<Vec<DynamicTensor>>,
        input_dims: Vec<usize>,
        batch_size: usize,
        ctx_pos: usize,
    ) -> (Vec<Vec<DynamicTensor>>, usize) {
        assert_eq!(
            streams.len(),
            1,
            "CombinerConnector backward: ожидается один поток градиентов"
        );

        let stream = &streams[0];
        let mut data = Vec::with_capacity(batch_size);
        for d in stream.iter() {
            if let DynamicTensor::Dim1(t) = d {
                data.push(t.data[0].clone());
            } else {
                panic!("CombinerConnector backward: ожидается Dim1");
            }
        }
        let delta = DynamicTensor::Dim1(Tensor2D::new(data));

        let ctx = DynamicContext::Ctx1D(
            crate::layers::context1d::LayerContext1D::CombinerConnector {
                inputs: vec![Tensor2D::zeros(1, 0); input_dims.len()],
            },
        );

        let connector = CombinerConnector::new(input_dims.clone());
        let (in_deltas, _) = connector.backward(&ctx, &delta);

        let num_inputs = in_deltas.len();
        let mut new_streams: Vec<Vec<DynamicTensor>> = Vec::with_capacity(num_inputs);
        for in_idx in 0..num_inputs {
            let mut stream = Vec::with_capacity(batch_size);
            if let DynamicTensor::Dim1(t) = &in_deltas[in_idx] {
                for r in 0..batch_size {
                    stream.push(DynamicTensor::Dim1(Tensor2D::new(vec![t.data[r].clone()])));
                }
            } else {
                panic!("CombinerConnector backward вернул не Dim1");
            }
            new_streams.push(stream);
        }

        (new_streams, ctx_pos - 1)
    }
}