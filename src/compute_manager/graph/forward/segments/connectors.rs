// src/compute_manager/graph/forward/segments/connectors.rs

use std::time::Instant;
use faer::Mat;
use crate::compute_manager::graph::model::MixedModel;
use crate::compute_manager::graph::types::DynamicContext;
use crate::device_plan::plan::ComputeDevice;
use crate::layers::splitter_connector::SplitterConnector;
use crate::layers::combiner_connector::CombinerConnector;
use crate::layers::splitter::Splitter;
use crate::layers::combiner::Combiner;
use crate::layers::mat_context::MatContext;

impl MixedModel {
    // ---------------------------------------------------------------
    // SplitterConnector (активный, два входа → два выхода)
    // ---------------------------------------------------------------
    pub(crate) fn process_splitter_connector_forward(
        &mut self,
        dim_a: usize,
        dim_b: usize,
        batch_size: usize,
        stream_matrices: &mut Vec<Mat<f32>>,
        all_ctxs: &mut Vec<Vec<DynamicContext>>,
        seg_index: usize,
    ) {
        let start = Instant::now();
        let device = self.segment_placement
            .get(seg_index)
            .map(|p| p.compute_device.clone())
            .unwrap_or(ComputeDevice::Cpu { id: 0, threads: 1 });

        assert_eq!(
            stream_matrices.len(),
            2,
            "SplitterConnector forward: expected 2 input streams"
        );

        let input_a_mat = stream_matrices[0].clone();
        let input_b_mat = stream_matrices[1].clone();

        let connector = SplitterConnector::new(dim_a, dim_b);
        let (out_a_mat, out_b_mat, ctx) = connector.forward_mat(&input_a_mat, &input_b_mat);

        // Сохраняем контекст для каждого сэмпла в батче
        for sample_ctxs in all_ctxs.iter_mut() {
            sample_ctxs.push(ctx.clone());
        }

        *stream_matrices = vec![out_a_mat, out_b_mat];

        let duration = start.elapsed().as_nanos() as f64;
        self.record_segment_timing(seg_index, &device, duration);
    }

    // ---------------------------------------------------------------
    // CombinerConnector (активный, N входов → N выходов, прозрачный)
    // ---------------------------------------------------------------
    pub(crate) fn process_combiner_connector_forward(
        &mut self,
        input_dims: Vec<usize>,
        batch_size: usize,
        stream_matrices: &mut Vec<Mat<f32>>,
        all_ctxs: &mut Vec<Vec<DynamicContext>>,
        seg_index: usize,
    ) {
        let start = Instant::now();
        let device = self.segment_placement
            .get(seg_index)
            .map(|p| p.compute_device.clone())
            .unwrap_or(ComputeDevice::Cpu { id: 0, threads: 1 });

        let n = input_dims.len();
        assert_eq!(
            stream_matrices.len(),
            n,
            "CombinerConnector forward: expected {} input streams, got {}",
            n,
            stream_matrices.len()
        );

        // Для каждого входного потока вызываем forward (identity) и сохраняем контекст
        // только для первого потока (как было раньше)
        for (stream_idx, matrix) in stream_matrices.iter().enumerate() {
            let connector = CombinerConnector::new(vec![]);
            let (_, ctx) = connector.forward_mat(matrix);

            if stream_idx == 0 {
                for sample_ctxs in all_ctxs.iter_mut() {
                    sample_ctxs.push(ctx.clone());
                }
            }
        }
        // stream_matrices остаются без изменений

        let duration = start.elapsed().as_nanos() as f64;
        self.record_segment_timing(seg_index, &device, duration);
    }

    // ---------------------------------------------------------------
    // Обучаемый Splitter (GPU + CPU)
    // ---------------------------------------------------------------
    pub(crate) fn process_splitter_forward(
        &mut self,
        input_dim: usize,
        output_dims: Vec<usize>,
        slice: crate::model_plan::param_store::ParamSlice,
        batch_size: usize,
        stream_matrices: &mut Vec<Mat<f32>>,
        all_ctxs: &mut Vec<Vec<DynamicContext>>,
        seg_index: usize,
    ) {
        let start = Instant::now();
        let device = self.segment_placement
            .get(seg_index)
            .map(|p| p.compute_device.clone())
            .unwrap_or(ComputeDevice::Cpu { id: 0, threads: 1 });

        assert_eq!(
            stream_matrices.len(),
            1,
            "Splitter forward: expected 1 input stream"
        );

        let input_mat = stream_matrices[0].clone();
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

            for sample_ctxs in all_ctxs.iter_mut() {
                sample_ctxs.push(ctx.clone());
            }

            *stream_matrices = vec![a_mat, b_mat];
        } else {
            // --- CPU путь ---
            let (a_mat, b_mat, pre_a_mat, pre_b_mat) =
                splitter.forward_mat(&input_mat, &params, &slice);

            let ctx = DynamicContext::Mat(MatContext::Splitter {
                input: input_mat.clone(),
                pre_a: pre_a_mat.clone(),
                pre_b: pre_b_mat.clone(),
            });

            for sample_ctxs in all_ctxs.iter_mut() {
                sample_ctxs.push(ctx.clone());
            }

            *stream_matrices = vec![a_mat, b_mat];
        }

        let duration = start.elapsed().as_nanos() as f64;
        self.record_segment_timing(seg_index, &device, duration);
    }

    // ---------------------------------------------------------------
    // Обучаемый Combiner (GPU + CPU)
    // ---------------------------------------------------------------
    pub(crate) fn process_combiner_forward(
        &mut self,
        input_dim: usize,
        output_dim: usize,
        slice: crate::model_plan::param_store::ParamSlice,
        batch_size: usize,
        stream_matrices: &mut Vec<Mat<f32>>,
        all_ctxs: &mut Vec<Vec<DynamicContext>>,
        seg_index: usize,
    ) {
        let start = Instant::now();
        let device = self.segment_placement
            .get(seg_index)
            .map(|p| p.compute_device.clone())
            .unwrap_or(ComputeDevice::Cpu { id: 0, threads: 1 });

        assert_eq!(
            stream_matrices.len(),
            2,
            "Combiner forward: expected 2 input streams"
        );

        let a_mat = stream_matrices[0].clone();
        let b_mat = stream_matrices[1].clone();

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

            for sample_ctxs in all_ctxs.iter_mut() {
                sample_ctxs.push(ctx.clone());
            }

            *stream_matrices = vec![out_mat];
        } else {
            // --- CPU путь ---
            let out_mat = combiner.forward_mat(&a_mat, &b_mat, &params, &slice);

            let ctx = DynamicContext::Mat(MatContext::Combiner {
                input_a: a_mat.clone(),
                input_b: b_mat.clone(),
                pre_act: Mat::zeros(batch_size, output_dim),
            });

            for sample_ctxs in all_ctxs.iter_mut() {
                sample_ctxs.push(ctx.clone());
            }

            *stream_matrices = vec![out_mat];
        }

        let duration = start.elapsed().as_nanos() as f64;
        self.record_segment_timing(seg_index, &device, duration);
    }
}