// src/compute_manager/graph/forward/segments/connectors.rs

use crate::tensor::Tensor2D;
use crate::compute_manager::dim_change::DynamicTensor;
use crate::compute_manager::graph::model::MixedModel;
use crate::compute_manager::graph::types::DynamicContext;
use crate::layers::splitter_connector::SplitterConnector;
use crate::layers::combiner_connector::CombinerConnector;
use crate::layers::splitter::Splitter;
use crate::layers::combiner::Combiner;
use crate::linalg;
use crate::layers::mat_context::MatContext;
use faer::Mat;

impl MixedModel {
    // ---------------------------------------------------------------
    // SplitterConnector (активный, два входа → два выхода)
    // ---------------------------------------------------------------
    pub(crate) fn process_splitter_connector_forward(
        &self,
        dim_a: usize,
        dim_b: usize,
        batch_size: usize,
        streams: &mut Vec<Vec<DynamicTensor>>,
        all_ctxs: &mut Vec<Vec<DynamicContext>>,
    ) {
        assert_eq!(streams.len(), 2, "SplitterConnector forward: expected 2 input streams");

        let stream_a = &streams[0];
        let stream_b = &streams[1];

        let mut data_a = Vec::with_capacity(batch_size);
        let mut data_b = Vec::with_capacity(batch_size);

        for (d_a, d_b) in stream_a.iter().zip(stream_b.iter()) {
            if let (DynamicTensor::Dim1(t_a), DynamicTensor::Dim1(t_b)) = (d_a, d_b) {
                data_a.push(t_a.data[0].clone());
                data_b.push(t_b.data[0].clone());
            } else {
                panic!("SplitterConnector forward: expected Dim1 inputs");
            }
        }

        let input_a_mat = linalg::tensor2d_to_faer(&Tensor2D::new(data_a));
        let input_b_mat = linalg::tensor2d_to_faer(&Tensor2D::new(data_b));

        let connector = SplitterConnector::new(dim_a, dim_b);
        let (out_a_mat, out_b_mat, ctx) = connector.forward_mat(&input_a_mat, &input_b_mat);

        let out_a_tensor = linalg::faer_to_tensor2d(&out_a_mat);
        let out_b_tensor = linalg::faer_to_tensor2d(&out_b_mat);

        let mut new_stream_a = Vec::with_capacity(batch_size);
        let mut new_stream_b = Vec::with_capacity(batch_size);

        for r in 0..batch_size {
            new_stream_a.push(DynamicTensor::Dim1(Tensor2D::new(vec![out_a_tensor.data[r].clone()])));
            new_stream_b.push(DynamicTensor::Dim1(Tensor2D::new(vec![out_b_tensor.data[r].clone()])));
        }

        for s in 0..batch_size {
            all_ctxs[s].push(ctx.clone());
        }

        *streams = vec![new_stream_a, new_stream_b];
    }

    // ---------------------------------------------------------------
    // CombinerConnector (активный, N входов → N выходов, прозрачный)
    // ---------------------------------------------------------------
    pub(crate) fn process_combiner_connector_forward(
        &self,
        input_dims: Vec<usize>,
        batch_size: usize,
        streams: &mut Vec<Vec<DynamicTensor>>,
        all_ctxs: &mut Vec<Vec<DynamicContext>>,
    ) {
        let n = input_dims.len();
        assert_eq!(streams.len(), n, "CombinerConnector forward: expected {} input streams, got {}", n, streams.len());

        let mut out_streams: Vec<Vec<DynamicTensor>> = Vec::with_capacity(n);

        for (stream_idx, stream) in streams.iter().enumerate() {
            let mut data = Vec::with_capacity(batch_size);
            for d in stream.iter() {
                if let DynamicTensor::Dim1(t) = d {
                    data.push(t.data[0].clone());
                } else {
                    panic!("CombinerConnector forward: expected Dim1 in stream {}", stream_idx);
                }
            }

            let input_mat = linalg::tensor2d_to_faer(&Tensor2D::new(data));

            let connector = CombinerConnector::new(vec![]);
            let (output_mat, ctx) = connector.forward_mat(&input_mat);

            let out_tensor = linalg::faer_to_tensor2d(&output_mat);
            let mut new_stream = Vec::with_capacity(batch_size);
            for r in 0..batch_size {
                new_stream.push(DynamicTensor::Dim1(Tensor2D::new(vec![out_tensor.data[r].clone()])));
            }

            if stream_idx == 0 {
                for s in 0..batch_size {
                    all_ctxs[s].push(ctx.clone());
                }
            }
            out_streams.push(new_stream);
        }

        *streams = out_streams;
    }

    // ---------------------------------------------------------------
    // Обучаемый Splitter (GPU + CPU)
    // ---------------------------------------------------------------
    pub(crate) fn process_splitter_forward(
        &self,
        input_dim: usize,
        output_dims: Vec<usize>,
        slice: crate::model_plan::param_store::ParamSlice,
        batch_size: usize,
        streams: &mut Vec<Vec<DynamicTensor>>,
        all_ctxs: &mut Vec<Vec<DynamicContext>>,
    ) {
        assert_eq!(streams.len(), 1, "Splitter forward: expected 1 input stream");

        let input_stream = &streams[0];
        let mut data = Vec::with_capacity(batch_size);
        for d in input_stream.iter() {
            if let DynamicTensor::Dim1(t) = d {
                data.push(t.data[0].clone());
            } else {
                panic!("Splitter forward: expected Dim1");
            }
        }
        let input_mat = linalg::tensor2d_to_faer(&Tensor2D::new(data));

        let params = self.store.lock().unwrap().all_params();
        let splitter = Splitter::new(input_dim, output_dims.clone());
        let (wa, wb, bias_a, bias_b) = splitter.get_weights_and_biases(&params, &slice);

        if let Some(ref gpu_compute_mutex) = self.gpu_compute {
            // --- GPU путь ---
            let gpu = gpu_compute_mutex.lock().unwrap();
            let (a_mat, b_mat, pre_a_mat, pre_b_mat) =
                gpu.run_splitter_forward(&input_mat, &wa, &bias_a, &wb, &bias_b);

            let ctx = DynamicContext::Mat(MatContext::Splitter {
                input: input_mat.clone(),
                pre_a: pre_a_mat.clone(),
                pre_b: pre_b_mat.clone(),
            });

            let a_tensor = linalg::faer_to_tensor2d(&a_mat);
            let b_tensor = linalg::faer_to_tensor2d(&b_mat);

            let mut stream_a = Vec::with_capacity(batch_size);
            let mut stream_b = Vec::with_capacity(batch_size);
            for r in 0..batch_size {
                stream_a.push(DynamicTensor::Dim1(Tensor2D::new(vec![a_tensor.data[r].clone()])));
                stream_b.push(DynamicTensor::Dim1(Tensor2D::new(vec![b_tensor.data[r].clone()])));
            }
            for s in 0..batch_size {
                all_ctxs[s].push(ctx.clone());
            }
            *streams = vec![stream_a, stream_b];
        } else {
            // --- CPU путь (оригинальный) ---
            let (a_mat, b_mat, pre_a_mat, pre_b_mat) = splitter.forward_mat(&input_mat, &params, &slice);

            let ctx = DynamicContext::Mat(MatContext::Splitter {
                input: input_mat.clone(),
                pre_a: pre_a_mat.clone(),
                pre_b: pre_b_mat.clone(),
            });

            let a_tensor = linalg::faer_to_tensor2d(&a_mat);
            let b_tensor = linalg::faer_to_tensor2d(&b_mat);

            let mut stream_a = Vec::with_capacity(batch_size);
            let mut stream_b = Vec::with_capacity(batch_size);
            for r in 0..batch_size {
                stream_a.push(DynamicTensor::Dim1(Tensor2D::new(vec![a_tensor.data[r].clone()])));
                stream_b.push(DynamicTensor::Dim1(Tensor2D::new(vec![b_tensor.data[r].clone()])));
            }
            for s in 0..batch_size {
                all_ctxs[s].push(ctx.clone());
            }
            *streams = vec![stream_a, stream_b];
        }
    }

    // ---------------------------------------------------------------
    // Обучаемый Combiner (GPU + CPU)
    // ---------------------------------------------------------------
    pub(crate) fn process_combiner_forward(
        &self,
        input_dim: usize,
        output_dim: usize,
        slice: crate::model_plan::param_store::ParamSlice,
        batch_size: usize,
        streams: &mut Vec<Vec<DynamicTensor>>,
        all_ctxs: &mut Vec<Vec<DynamicContext>>,
    ) {
        assert_eq!(streams.len(), 2, "Combiner forward: expected 2 input streams");

        let mut data_a = Vec::with_capacity(batch_size);
        let mut data_b = Vec::with_capacity(batch_size);
        for (d_a, d_b) in streams[0].iter().zip(streams[1].iter()) {
            if let (DynamicTensor::Dim1(t_a), DynamicTensor::Dim1(t_b)) = (d_a, d_b) {
                data_a.push(t_a.data[0].clone());
                data_b.push(t_b.data[0].clone());
            } else {
                panic!("Combiner forward: expected Dim1");
            }
        }

        let a_mat = linalg::tensor2d_to_faer(&Tensor2D::new(data_a));
        let b_mat = linalg::tensor2d_to_faer(&Tensor2D::new(data_b));

        let params = self.store.lock().unwrap().all_params();
        let combiner = Combiner::new(vec![input_dim, input_dim], output_dim);
        let (wa, wb, bias) = combiner.get_weights_and_bias(&params, &slice);

        if let Some(ref gpu_compute_mutex) = self.gpu_compute {
            // --- GPU путь ---
            let gpu = gpu_compute_mutex.lock().unwrap();
            let (out_mat, pre_mat) = gpu.run_combiner_forward(&a_mat, &b_mat, &wa, &wb, &bias);

            let ctx = DynamicContext::Mat(MatContext::Combiner {
                input_a: a_mat.clone(),
                input_b: b_mat.clone(),
                pre_act: pre_mat.clone(),
            });

            let out_tensor = linalg::faer_to_tensor2d(&out_mat);
            let mut combined_stream = Vec::with_capacity(batch_size);
            for r in 0..batch_size {
                combined_stream.push(DynamicTensor::Dim1(Tensor2D::new(vec![out_tensor.data[r].clone()])));
            }
            for s in 0..batch_size {
                all_ctxs[s].push(ctx.clone());
            }
            *streams = vec![combined_stream];
        } else {
            // --- CPU путь (оригинальный) ---
            let out_mat = combiner.forward_mat(&a_mat, &b_mat, &params, &slice);
            let ctx = DynamicContext::Mat(MatContext::Combiner {
                input_a: a_mat.clone(),
                input_b: b_mat.clone(),
                pre_act: Mat::zeros(batch_size, output_dim),
            });

            let out_tensor = linalg::faer_to_tensor2d(&out_mat);
            let mut combined_stream = Vec::with_capacity(batch_size);
            for r in 0..batch_size {
                combined_stream.push(DynamicTensor::Dim1(Tensor2D::new(vec![out_tensor.data[r].clone()])));
            }
            for s in 0..batch_size {
                all_ctxs[s].push(ctx.clone());
            }
            *streams = vec![combined_stream];
        }
    }
}

fn mat_to_flat(mat: &faer::Mat<f32>) -> Vec<f32> {
    let rows = mat.nrows();
    let cols = mat.ncols();
    let mut flat = Vec::with_capacity(rows * cols);
    for i in 0..rows {
        for j in 0..cols {
            flat.push(mat[(i, j)]);
        }
    }
    flat
}