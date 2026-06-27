// src/compute_manager/graph/backward/main.rs

use faer::Mat;
use crate::compute_manager::dim_change::DynamicTensor;
use crate::compute_manager::graph::model::MixedModel;
use crate::compute_manager::graph::types::{DynamicContext, Segment};
use crate::layers::UniversalLayer;
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

        for seg in self.segments.iter().rev() {
            match seg {
                Segment::Unsqueeze(target_dims) => {
                    self.process_unsqueeze_backward(&mut streams, target_dims);
                }
                Segment::ReduceMean(target_dims) => {
                    self.process_reduce_mean_backward(&mut streams, target_dims);
                }
                Segment::UniversalProcessor(proc, slices, stream_indices) => {
                    // Обратный проход UniversalProcessor пока всегда на CPU.
                    // GPU-реализация backward будет добавлена позднее.
                    let result = self.process_universal_processor_backward_mat(
                        proc, slices, &streams, contexts, ctx_pos, &params, &mut total_grad, stream_indices,
                    );
                    streams = result.0;
                    ctx_pos = result.1;
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
                    let (x_tensor, pre_a_flat, pre_b_flat) = match ctx {
                        DynamicContext::Ctx1D(c) => match c {
                            crate::layers::context1d::LayerContext1D::Splitter { input, pre_a, pre_b } =>
                                (input, pre_a.clone(), pre_b.clone()),
                            _ => panic!("Expected Splitter context"),
                        },
                        _ => panic!("Expected Ctx1D"),
                    };

                    let da_tensor = DynamicTensor::Dim1(crate::tensor::Tensor2D::new(
                        streams[0].iter().map(|d| match d { DynamicTensor::Dim1(t) => t.data[0].clone(), _ => panic!() }).collect()
                    ));
                    let db_tensor = DynamicTensor::Dim1(crate::tensor::Tensor2D::new(
                        streams[1].iter().map(|d| match d { DynamicTensor::Dim1(t) => t.data[0].clone(), _ => panic!() }).collect()
                    ));

                    let x_mat = linalg::tensor2d_to_faer(x_tensor);
                    let da_mat = linalg::tensor2d_to_faer(&match da_tensor { DynamicTensor::Dim1(t) => t, _ => unreachable!() });
                    let db_mat = linalg::tensor2d_to_faer(&match db_tensor { DynamicTensor::Dim1(t) => t, _ => unreachable!() });
                    let batch = x_mat.nrows();
                    let p = output_dims[0];
                    let q = output_dims[1];
                    let pre_a_mat = flat_to_mat(pre_a_flat, batch, p);
                    let pre_b_mat = flat_to_mat(pre_b_flat, batch, q);
                    let (wa, wb, _, _) = crate::layers::Splitter::new(*input_dim, output_dims.clone()).get_weights_and_biases(&params, slice);
                    let (dx_mat, grad) = crate::layers::Splitter::new(*input_dim, output_dims.clone()).backward_mat(
                        &x_mat, &da_mat, &db_mat, &pre_a_mat, &pre_b_mat, &wa, &wb,
                    );

                    for (idx, &g) in grad.iter().enumerate() {
                        total_grad[slice.start + idx] += g;
                    }

                    let dx_tensor = linalg::faer_to_tensor2d(&dx_mat);
                    let mut combined_stream = Vec::with_capacity(streams[0].len());
                    for r in 0..streams[0].len() {
                        combined_stream.push(DynamicTensor::Dim1(crate::tensor::Tensor2D::new(vec![dx_tensor.data[r].clone()])));
                    }
                    streams = vec![combined_stream];
                    ctx_pos -= 1;
                }
                Segment::Combiner { input_dim, output_dim, slice } => {
                    assert!(ctx_pos > 0, "Backward: no context for Combiner");
                    let ctx = &contexts[0][ctx_pos - 1];
                    let (a_tensor, b_tensor) = match ctx {
                        DynamicContext::Ctx1D(c) => match c {
                            crate::layers::context1d::LayerContext1D::Combiner { input_a, input_b, .. } =>
                                (input_a, input_b),
                            _ => panic!("Expected Combiner context"),
                        },
                        _ => panic!("Expected Ctx1D"),
                    };

                    let delta_single = DynamicTensor::Dim1(crate::tensor::Tensor2D::new(
                        streams[0].iter().map(|d| match d { DynamicTensor::Dim1(t) => t.data[0].clone(), _ => panic!() }).collect()
                    ));
                    let dout_mat = linalg::tensor2d_to_faer(&match delta_single { DynamicTensor::Dim1(t) => t, _ => unreachable!() });

                    let a_mat = linalg::tensor2d_to_faer(a_tensor);
                    let b_mat = linalg::tensor2d_to_faer(b_tensor);

                    let combiner = crate::layers::Combiner::new(vec![*input_dim, *input_dim], *output_dim);
                    let (da_mat, db_mat, grad) = combiner.backward_mat(&a_mat, &b_mat, &dout_mat, &params, slice);

                    for (idx, &g) in grad.iter().enumerate() {
                        total_grad[slice.start + idx] += g;
                    }

                    let da_tensor = linalg::faer_to_tensor2d(&da_mat);
                    let db_tensor = linalg::faer_to_tensor2d(&db_mat);
                    let mut stream_a = Vec::with_capacity(streams[0].len());
                    let mut stream_b = Vec::with_capacity(streams[0].len());
                    for r in 0..streams[0].len() {
                        stream_a.push(DynamicTensor::Dim1(crate::tensor::Tensor2D::new(vec![da_tensor.data[r].clone()])));
                        stream_b.push(DynamicTensor::Dim1(crate::tensor::Tensor2D::new(vec![db_tensor.data[r].clone()])));
                    }
                    streams = vec![stream_a, stream_b];
                    ctx_pos -= 1;
                }
            }
        }

        assert_eq!(streams.len(), self.input_stream_count,
            "backward_mat_multi: input stream count mismatch");

        let in_mats: Vec<Mat<f32>> = streams.iter()
            .map(|stream| {
                let first = stream.first().unwrap();
                let features = match first { DynamicTensor::Dim1(t) => t.dim2, _ => panic!() };
                let batch = stream.len();
                let mut mat = Mat::zeros(batch, features);
                for (i, sample) in stream.iter().enumerate() {
                    match sample {
                        DynamicTensor::Dim1(t) => {
                            for (j, &val) in t.data[0].iter().enumerate() {
                                mat[(i, j)] = val;
                            }
                        }
                        _ => panic!("Only Dim1 in backward"),
                    }
                }
                mat
            })
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
        batch_size: usize,
        ctx_pos: usize,
    ) -> (Vec<Vec<DynamicTensor>>, usize) {
        let stream_a = &streams[0];
        let stream_b = &streams[1];
        let mut data_a = Vec::with_capacity(batch_size);
        let mut data_b = Vec::with_capacity(batch_size);
        for (d_a, d_b) in stream_a.iter().zip(stream_b.iter()) {
            if let (DynamicTensor::Dim1(t_a), DynamicTensor::Dim1(t_b)) = (d_a, d_b) {
                data_a.push(t_a.data[0].clone());
                data_b.push(t_b.data[0].clone());
            } else { panic!("Expected Dim1"); }
        }
        let delta_a_mat = linalg::tensor2d_to_faer(&crate::tensor::Tensor2D::new(data_a));
        let delta_b_mat = linalg::tensor2d_to_faer(&crate::tensor::Tensor2D::new(data_b));
        let connector = crate::layers::SplitterConnector::new(dim_a, dim_b);
        let ctx = DynamicContext::Ctx1D(crate::layers::context1d::LayerContext1D::SplitterConnector {
            input: crate::tensor::Tensor2D::zeros(1, 0),
        });
        let (in_a_mat, in_b_mat, _) = connector.backward_mat(&ctx, &delta_a_mat, &delta_b_mat);
        let in_a_tensor = linalg::faer_to_tensor2d(&in_a_mat);
        let in_b_tensor = linalg::faer_to_tensor2d(&in_b_mat);
        let mut new_a = Vec::with_capacity(batch_size);
        let mut new_b = Vec::with_capacity(batch_size);
        for r in 0..batch_size {
            new_a.push(DynamicTensor::Dim1(crate::tensor::Tensor2D::new(vec![in_a_tensor.data[r].clone()])));
            new_b.push(DynamicTensor::Dim1(crate::tensor::Tensor2D::new(vec![in_b_tensor.data[r].clone()])));
        }
        (vec![new_a, new_b], ctx_pos - 1)
    }

    fn process_combiner_connector_backward_mat(
        &self,
        streams: &Vec<Vec<DynamicTensor>>,
        input_dims: Vec<usize>,
        batch_size: usize,
        ctx_pos: usize,
    ) -> (Vec<Vec<DynamicTensor>>, usize) {
        let n = input_dims.len();
        let mut out_streams = Vec::with_capacity(n);
        for (_i, stream) in streams.iter().enumerate() {
            let mut data = Vec::with_capacity(batch_size);
            for d in stream.iter() {
                if let DynamicTensor::Dim1(t) = d {
                    data.push(t.data[0].clone());
                } else { panic!("Expected Dim1"); }
            }
            let delta_mat = linalg::tensor2d_to_faer(&crate::tensor::Tensor2D::new(data));
            let connector = crate::layers::CombinerConnector::new(vec![]);
            let ctx = DynamicContext::Ctx1D(crate::layers::context1d::LayerContext1D::CombinerConnector {
                inputs: vec![crate::tensor::Tensor2D::zeros(1, 0)],
            });
            let (in_mat, _) = connector.backward_mat(&ctx, &delta_mat);
            let in_tensor = linalg::faer_to_tensor2d(&in_mat);
            let mut new_stream = Vec::with_capacity(batch_size);
            for r in 0..batch_size {
                new_stream.push(DynamicTensor::Dim1(crate::tensor::Tensor2D::new(vec![in_tensor.data[r].clone()])));
            }
            out_streams.push(new_stream);
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

fn flat_to_mat(flat: Vec<f32>, rows: usize, cols: usize) -> Mat<f32> {
    Mat::from_fn(rows, cols, |i, j| flat[i * cols + j])
}