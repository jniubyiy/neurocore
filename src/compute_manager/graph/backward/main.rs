// src/compute_manager/graph/backward/main.rs

use faer::Mat;
use crate::compute_manager::dim_change::DynamicTensor;
use crate::compute_manager::graph::model::MixedModel;
use crate::compute_manager::graph::types::{DynamicContext, Segment};
use crate::layers::UniversalLayer;
use crate::layers::mat_context::MatContext;
use crate::linalg;

impl MixedModel {
    pub fn backward_mat_multi(
        &self,
        contexts: &[Vec<DynamicContext>],
        deltas: &[Mat<f32>],
    ) -> (Vec<Mat<f32>>, Vec<Vec<f32>>) {
        assert_eq!(deltas.len(), self.output_stream_count,
            "backward_mat_multi: expected {} deltas, got {}", self.output_stream_count, deltas.len());

        let params = self.store.lock().unwrap().all_params().to_vec();
        let param_len = params.len();
        let mut total_grad = vec![0.0f32; param_len];

        let mut streams: Vec<Vec<DynamicTensor>> = deltas.iter().map(|delta| {
            let batch = delta.nrows();
            let cols = delta.ncols();
            (0..batch)
                .map(|r| {
                    let row: Vec<f32> = (0..cols).map(|c| delta[(r, c)]).collect();
                    DynamicTensor::Dim1(crate::tensor::Tensor2D::new(vec![row]))
                })
                .collect()
        }).collect();

        let total_context_len = contexts.first().map(|c| c.len()).unwrap_or(0);
        let mut ctx_pos = total_context_len;

        for (seg_idx, seg) in self.segments.iter().enumerate().rev() {
            match seg {
                Segment::Unsqueeze(target_dims) => {
                    self.process_unsqueeze_backward(&mut streams, target_dims);
                }
                Segment::ReduceMean(target_dims) => {
                    self.process_reduce_mean_backward(&mut streams, target_dims);
                }
                Segment::UniversalProcessor(proc, slices, stream_indices) => {
                    if let Some(ref gpu_compute_mutex) = self.gpu_compute {
                        let gpu_compute = gpu_compute_mutex.lock().unwrap();
                        let result = self.process_universal_processor_backward_gpu(
                            &gpu_compute,
                            proc,
                            slices,
                            &streams,
                            contexts,
                            ctx_pos,
                            &params,
                            &mut total_grad,
                            stream_indices,
                        );
                        streams = result.0;
                        ctx_pos = result.1;
                    } else {
                        let result = self.process_universal_processor_backward_mat(
                            proc, slices, &streams, contexts, ctx_pos, &params, &mut total_grad, stream_indices,
                        );
                        streams = result.0;
                        ctx_pos = result.1;
                    }
                }
                Segment::SplitterConnector { dim_a, dim_b } => {
                    let result = self.process_splitter_connector_backward_mat(
                        &streams, *dim_a, *dim_b, streams[0].len(), ctx_pos,
                    );
                    streams = result.0;
                    ctx_pos = result.1;
                }
                Segment::CombinerConnector { input_dims, .. } => {
                    let result = self.process_combiner_connector_backward_mat(
                        &streams, input_dims.clone(), streams[0].len(), ctx_pos,
                    );
                    streams = result.0;
                    ctx_pos = result.1;
                }
                Segment::Splitter { input_dim, output_dims, slice } => {
                    assert!(ctx_pos > 0, "Backward: no context for Splitter");
                    let ctx = &contexts[0][ctx_pos - 1];
                    let (x_mat, pre_a_mat, pre_b_mat) = match ctx {
                        DynamicContext::Mat(MatContext::Splitter { input, pre_a, pre_b }) =>
                            (input.clone(), pre_a.clone(), pre_b.clone()),
                        _ => panic!("Expected Splitter context"),
                    };

                    let da_mat = samples_to_mat(&streams[0]);
                    let db_mat = samples_to_mat(&streams[1]);

                    let (wa, wb, _, _) = crate::layers::Splitter::new(*input_dim, output_dims.clone()).get_weights_and_biases(&params, slice);
                    let (dx_mat, grad) = if let Some(ref gpu) = self.gpu_compute {
                        let gpu = gpu.lock().unwrap();
                        gpu.run_splitter_backward(&x_mat, &da_mat, &db_mat, &pre_a_mat, &pre_b_mat, &wa, &wb)
                    } else {
                        crate::layers::Splitter::new(*input_dim, output_dims.clone())
                            .backward_mat(&x_mat, &da_mat, &db_mat, &pre_a_mat, &pre_b_mat, &wa, &wb)
                    };

                    for (idx, &g) in grad.iter().enumerate() {
                        total_grad[slice.start + idx] += g;
                    }

                    let combined_stream = mat_to_samples(&dx_mat);
                    streams = vec![combined_stream];
                    ctx_pos -= 1;
                }
                Segment::Combiner { input_dim, output_dim, slice } => {
                    assert!(ctx_pos > 0, "Backward: no context for Combiner");
                    let ctx = &contexts[0][ctx_pos - 1];
                    let (a_mat, b_mat, pre_mat) = match ctx {
                        DynamicContext::Mat(MatContext::Combiner { input_a, input_b, pre_act }) =>
                            (input_a.clone(), input_b.clone(), pre_act.clone()),
                        _ => panic!("Expected Combiner context"),
                    };

                    let dout_mat = samples_to_mat(&streams[0]);

                    let combiner = crate::layers::Combiner::new(vec![*input_dim, *input_dim], *output_dim);
                    let (wa, wb, _) = combiner.get_weights_and_bias(&params, slice);
                    let (da_mat, db_mat, grad) = if let Some(ref gpu) = self.gpu_compute {
                        let gpu = gpu.lock().unwrap();
                        gpu.run_combiner_backward(&a_mat, &b_mat, &dout_mat, &pre_mat, &wa, &wb)
                    } else {
                        combiner.backward_mat(&a_mat, &b_mat, &dout_mat, &params, slice)
                    };

                    for (idx, &g) in grad.iter().enumerate() {
                        total_grad[slice.start + idx] += g;
                    }

                    let stream_a = mat_to_samples(&da_mat);
                    let stream_b = mat_to_samples(&db_mat);
                    streams = vec![stream_a, stream_b];
                    ctx_pos -= 1;
                }
            }
        }

        assert_eq!(streams.len(), self.input_stream_count,
            "backward_mat_multi: input stream count mismatch");

        let in_mats: Vec<Mat<f32>> = streams.iter()
            .map(|stream| samples_to_mat(stream))
            .collect();

        (in_mats, vec![total_grad])
    }

    pub fn backward_mat(
        &self,
        contexts: &[Vec<DynamicContext>],
        delta: &Mat<f32>,
    ) -> (Mat<f32>, Vec<Vec<f32>>) {
        let (ins, grads) = self.backward_mat_multi(contexts, &[delta.clone()]);
        assert_eq!(ins.len(), 1);
        (ins.into_iter().next().unwrap(), grads)
    }

    fn process_universal_processor_backward_gpu(
        &self,
        gpu_compute: &crate::compute_manager::gpu::GpuCompute,
        proc: &std::sync::Arc<Vec<Box<dyn UniversalLayer>>>,
        slices: &[crate::model_plan::param_store::ParamSlice],
        streams: &Vec<Vec<DynamicTensor>>,
        contexts: &[Vec<DynamicContext>],
        ctx_pos: usize,
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
        for &stream_idx in &active_indices {
            let stream_samples = &streams[stream_idx];
            let delta_mat = samples_to_mat(stream_samples);
            let pos_in_sorted = active_indices.iter().position(|&x| x == stream_idx).unwrap();
            let stream_ctx_start = ctx_pos - (active_indices.len() - pos_in_sorted) * num_layers;
            let layer_ctxs = &contexts[0][stream_ctx_start..stream_ctx_start + num_layers];

            let in_delta_mat = crate::compute_manager::gpu::processor::process_backward_gpu(
                gpu_compute,
                proc,
                slices,
                layer_ctxs,
                params,
                &delta_mat,
                total_grad,
            );

            let new_samples = mat_to_samples(&in_delta_mat);
            new_streams[stream_idx] = Some(new_samples);
        }
        for (i, opt) in new_streams.iter_mut().enumerate() {
            if opt.is_none() { *opt = Some(streams[i].clone()); }
        }
        let new_ctx_pos = ctx_pos - num_layers * active_indices.len();
        (new_streams.into_iter().map(|o| o.unwrap()).collect(), new_ctx_pos)
    }

    fn process_universal_processor_backward_mat(
        &self,
        proc: &std::sync::Arc<Vec<Box<dyn UniversalLayer>>>,
        slices: &[crate::model_plan::param_store::ParamSlice],
        streams: &Vec<Vec<DynamicTensor>>,
        contexts: &[Vec<DynamicContext>],
        ctx_pos: usize,
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
        for &stream_idx in &active_indices {
            let stream_samples = &streams[stream_idx];
            let delta_mat = samples_to_mat(stream_samples);
            let pos_in_sorted = active_indices.iter().position(|&x| x == stream_idx).unwrap();
            let stream_ctx_start = ctx_pos - (active_indices.len() - pos_in_sorted) * num_layers;
            let layer_ctxs: Vec<&DynamicContext> = contexts[0][stream_ctx_start..stream_ctx_start + num_layers].iter().collect();
            let (in_delta_mat, local_grad) = backward_universal_batch_mat(proc, slices, &layer_ctxs, &delta_mat, params);
            let new_samples = mat_to_samples(&in_delta_mat);
            new_streams[stream_idx] = Some(new_samples);
            for (idx, &g) in local_grad.iter().enumerate() {
                total_grad[idx] += g;
            }
        }
        for (i, opt) in new_streams.iter_mut().enumerate() {
            if opt.is_none() { *opt = Some(streams[i].clone()); }
        }
        let new_ctx_pos = ctx_pos - num_layers * active_indices.len();
        (new_streams.into_iter().map(|o| o.unwrap()).collect(), new_ctx_pos)
    }

    fn process_splitter_connector_backward_mat(
        &self,
        streams: &Vec<Vec<DynamicTensor>>,
        dim_a: usize,
        dim_b: usize,
        _batch_size: usize,
        ctx_pos: usize,
    ) -> (Vec<Vec<DynamicTensor>>, usize) {
        let stream_a = &streams[0];
        let stream_b = &streams[1];
        let delta_a_mat = samples_to_mat(stream_a);
        let delta_b_mat = samples_to_mat(stream_b);

        let connector = crate::layers::SplitterConnector::new(dim_a, dim_b);
        let ctx = DynamicContext::Mat(MatContext::SplitterConnector {
            input: Mat::zeros(0, 0),
        });
        let (in_a_mat, in_b_mat, _) = connector.backward_mat(&ctx, &delta_a_mat, &delta_b_mat);

        let new_a = mat_to_samples(&in_a_mat);
        let new_b = mat_to_samples(&in_b_mat);
        (vec![new_a, new_b], ctx_pos - 1)
    }

    fn process_combiner_connector_backward_mat(
        &self,
        streams: &Vec<Vec<DynamicTensor>>,
        input_dims: Vec<usize>,
        _batch_size: usize,
        ctx_pos: usize,
    ) -> (Vec<Vec<DynamicTensor>>, usize) {
        let n = input_dims.len();
        let mut out_streams = Vec::with_capacity(n);
        for stream in streams.iter() {
            let delta_mat = samples_to_mat(stream);
            let connector = crate::layers::CombinerConnector::new(vec![]);
            let ctx = DynamicContext::Mat(MatContext::CombinerConnector {
                inputs: vec![Mat::zeros(0, 0)],
            });
            let (in_mat, _) = connector.backward_mat(&ctx, &delta_mat);
            out_streams.push(mat_to_samples(&in_mat));
        }
        (out_streams, ctx_pos - 1)
    }
}

// Вспомогательные функции
fn samples_to_mat(samples: &[DynamicTensor]) -> Mat<f32> {
    let first = &samples[0];
    let features = match first { DynamicTensor::Dim1(t) => t.dim2, _ => panic!("Only Dim1") };
    let batch = samples.len();
    let mut mat = Mat::zeros(batch, features);
    for (i, sample) in samples.iter().enumerate() {
        match sample {
            DynamicTensor::Dim1(t) => { for (j, &val) in t.data[0].iter().enumerate() { mat[(i, j)] = val; } }
            _ => panic!("Only Dim1"),
        }
    }
    mat
}

fn mat_to_samples(mat: &Mat<f32>) -> Vec<DynamicTensor> {
    let batch = mat.nrows();
    let features = mat.ncols();
    let mut samples = Vec::with_capacity(batch);
    for i in 0..batch {
        let row: Vec<f32> = (0..features).map(|j| mat[(i, j)]).collect();
        samples.push(DynamicTensor::Dim1(crate::tensor::Tensor2D::new(vec![row])));
    }
    samples
}

fn backward_universal_batch_mat(
    layers: &[Box<dyn UniversalLayer>],
    slices: &[crate::model_plan::param_store::ParamSlice],
    ctxs: &[&DynamicContext],
    delta: &Mat<f32>,
    params: &[f32],
) -> (Mat<f32>, Vec<f32>) {
    let mut current_delta = delta.clone();
    let mut total_grad = vec![0.0f32; params.len()];
    for i in (0..layers.len()).rev() {
        let (in_delta, grad) = layers[i].backward_mat(ctxs[i], &current_delta, params, &slices[i]);
        current_delta = in_delta;
        for (idx, &g) in grad.iter().enumerate() {
            total_grad[idx] += g;
        }
    }
    (current_delta, total_grad)
}