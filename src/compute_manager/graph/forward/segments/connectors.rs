// src/compute_manager/graph/forward/segments/connectors.rs

use std::time::Instant;
use faer::Mat;
use crate::compute_manager::graph::model::MixedModel;
use crate::compute_manager::graph::types::DynamicContext;
use crate::compute_manager::matrix_buffer::{MatrixBuffer, TempMatrixPool};
use crate::device_plan::plan::ComputeDevice;
use crate::layers::splitter_connector::SplitterConnector;
use crate::layers::combiner_connector::CombinerConnector;
use crate::layers::splitter::Splitter;
use crate::layers::combiner::Combiner;
use crate::layers::mat_context::MatContext;
use crate::model_plan::param_store::ParamSlice;

impl MixedModel {
    // ---------------------------------------------------------------
    // SplitterConnector (активный, два входа → два выхода)
    // Старая версия (Mat<f32>)
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

        for sample_ctxs in all_ctxs.iter_mut() {
            sample_ctxs.push(ctx.clone());
        }

        *stream_matrices = vec![out_a_mat, out_b_mat];

        let duration = start.elapsed().as_nanos() as f64;
        self.record_segment_timing(seg_index, &device, duration);
    }

    // ---------------------------------------------------------------
    // CombinerConnector (активный, N входов → N выходов, прозрачный)
    // Старая версия (Mat<f32>)
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

        for (stream_idx, matrix) in stream_matrices.iter().enumerate() {
            let connector = CombinerConnector::new(vec![]);
            let (_, ctx) = connector.forward_mat(matrix);

            if stream_idx == 0 {
                for sample_ctxs in all_ctxs.iter_mut() {
                    sample_ctxs.push(ctx.clone());
                }
            }
        }

        let duration = start.elapsed().as_nanos() as f64;
        self.record_segment_timing(seg_index, &device, duration);
    }

    // ---------------------------------------------------------------
    // Обучаемый Splitter (GPU + CPU)
    // Старая версия (Mat<f32>)
    // ---------------------------------------------------------------
    pub(crate) fn process_splitter_forward(
        &mut self,
        input_dim: usize,
        output_dims: Vec<usize>,
        slice: ParamSlice,
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
    // Старая версия (Mat<f32>)
    // ---------------------------------------------------------------
    pub(crate) fn process_combiner_forward(
        &mut self,
        input_dim: usize,
        output_dim: usize,
        slice: ParamSlice,
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

    // ===================================================================
    // НОВЫЕ БУФЕРИЗОВАННЫЕ ВЕРСИИ ДЛЯ РАБОТЫ С MatrixBuffer
    // ===================================================================

    /// SplitterConnector с управляемыми буферами.
    /// Входные буферы забираются по владению, выходные создаются из пула.
    /// Контекст пока хранит копии Mat<f32> для совместимости.
    pub(crate) fn process_splitter_connector_forward_buffered(
        &mut self,
        pool: &mut TempMatrixPool,
        dim_a: usize,
        dim_b: usize,
        batch_size: usize,
        stream_buffers: &mut Vec<MatrixBuffer>,
        all_ctxs: &mut Vec<Vec<DynamicContext>>,
        seg_index: usize,
    ) {
        let start = Instant::now();
        let device = self.segment_placement
            .get(seg_index)
            .map(|p| p.compute_device.clone())
            .unwrap_or(ComputeDevice::Cpu { id: 0, threads: 1 });

        assert_eq!(stream_buffers.len(), 2, "SplitterConnector buffered: expected 2 input streams");

        // Извлекаем входные буферы, оставляя временные заглушки (будут перезаписаны)
        let input_a = std::mem::replace(&mut stream_buffers[0], MatrixBuffer::dummy(pool));
        let input_b = std::mem::replace(&mut stream_buffers[1], MatrixBuffer::dummy(pool));

        // Создаём выходные буферы того же размера, что и входные
        let rows_a = input_a.rows();
        let cols_a = input_a.cols();
        let rows_b = input_b.rows();
        let cols_b = input_b.cols();
        let mut out_a = pool.acquire(rows_a, cols_a);
        let mut out_b = pool.acquire(rows_b, cols_b);

        // Копируем данные в выходные буферы (identity для коннектора)
        out_a.as_mat_mut().copy_from(&input_a.as_mat());
        out_b.as_mat_mut().copy_from(&input_b.as_mat());

        // Строим контекст из Mat (временное копирование, пока MatContext не переведён)
        let ctx = DynamicContext::Mat(MatContext::SplitterConnector {
            input: input_a.as_mat().to_owned(),
        });

        for sample_ctxs in all_ctxs.iter_mut() {
            sample_ctxs.push(ctx.clone());
        }

        // Возвращаем входные буферы в пул
        pool.release(input_a);
        pool.release(input_b);

        *stream_buffers = vec![out_a, out_b];

        let duration = start.elapsed().as_nanos() as f64;
        self.record_segment_timing(seg_index, &device, duration);
    }

    /// CombinerConnector с управляемыми буферами.
    /// Все входные буферы остаются без изменений (прозрачный проход), только контекст.
    pub(crate) fn process_combiner_connector_forward_buffered(
        &mut self,
        pool: &mut TempMatrixPool,
        input_dims: Vec<usize>,
        batch_size: usize,
        stream_buffers: &mut Vec<MatrixBuffer>,
        all_ctxs: &mut Vec<Vec<DynamicContext>>,
        seg_index: usize,
    ) {
        let start = Instant::now();
        let device = self.segment_placement
            .get(seg_index)
            .map(|p| p.compute_device.clone())
            .unwrap_or(ComputeDevice::Cpu { id: 0, threads: 1 });

        let n = input_dims.len();
        assert_eq!(stream_buffers.len(), n,
            "CombinerConnector buffered: expected {} input streams, got {}",
            n, stream_buffers.len());

        // Сохраняем контекст только для первого потока (как в старом коде)
        for (stream_idx, buf) in stream_buffers.iter().enumerate() {
            if stream_idx == 0 {
                let connector = CombinerConnector::new(vec![]);
                let (_, ctx) = connector.forward_mat(&buf.as_mat().to_owned());
                for sample_ctxs in all_ctxs.iter_mut() {
                    sample_ctxs.push(ctx.clone());
                }
            }
        }

        let duration = start.elapsed().as_nanos() as f64;
        self.record_segment_timing(seg_index, &device, duration);
    }

    /// Обучаемый Splitter с управляемыми буферами (только CPU).
    pub(crate) fn process_splitter_forward_buffered(
        &mut self,
        pool: &mut TempMatrixPool,
        input_dim: usize,
        output_dims: Vec<usize>,
        slice: ParamSlice,
        batch_size: usize,
        stream_buffers: &mut Vec<MatrixBuffer>,
        all_ctxs: &mut Vec<Vec<DynamicContext>>,
        seg_index: usize,
    ) {
        let start = Instant::now();
        let device = self.segment_placement
            .get(seg_index)
            .map(|p| p.compute_device.clone())
            .unwrap_or(ComputeDevice::Cpu { id: 0, threads: 1 });

        assert_eq!(stream_buffers.len(), 1, "Splitter buffered: expected 1 input stream");

        let input_buf = std::mem::replace(&mut stream_buffers[0], MatrixBuffer::dummy(pool));
        let batch = input_buf.rows();
        let params = self.store.lock().unwrap().all_params();
        let splitter = Splitter::new(input_dim, output_dims.clone());
        let (wa, wb, bias_a, bias_b) = splitter.get_weights_and_biases(&params, &slice);

        // Пока выполняем на CPU (GPU-путь будет добавлен позже)
        let input_mat = input_buf.as_mat().to_owned();
        let (a_mat, b_mat, pre_a_mat, pre_b_mat) =
            splitter.forward_mat(&input_mat, &params, &slice);

        let mut out_a = pool.acquire(batch, output_dims[0]);
        let mut out_b = pool.acquire(batch, output_dims[1]);
        out_a.as_mat_mut().copy_from(&a_mat);
        out_b.as_mat_mut().copy_from(&b_mat);

        let ctx = DynamicContext::Mat(MatContext::Splitter {
            input: input_mat.clone(),
            pre_a: pre_a_mat.clone(),
            pre_b: pre_b_mat.clone(),
        });

        for sample_ctxs in all_ctxs.iter_mut() {
            sample_ctxs.push(ctx.clone());
        }

        pool.release(input_buf);

        *stream_buffers = vec![out_a, out_b];

        let duration = start.elapsed().as_nanos() as f64;
        self.record_segment_timing(seg_index, &device, duration);
    }

    /// Обучаемый Combiner с управляемыми буферами (только CPU).
    pub(crate) fn process_combiner_forward_buffered(
        &mut self,
        pool: &mut TempMatrixPool,
        input_dim: usize,
        output_dim: usize,
        slice: ParamSlice,
        batch_size: usize,
        stream_buffers: &mut Vec<MatrixBuffer>,
        all_ctxs: &mut Vec<Vec<DynamicContext>>,
        seg_index: usize,
    ) {
        let start = Instant::now();
        let device = self.segment_placement
            .get(seg_index)
            .map(|p| p.compute_device.clone())
            .unwrap_or(ComputeDevice::Cpu { id: 0, threads: 1 });

        assert_eq!(stream_buffers.len(), 2, "Combiner buffered: expected 2 input streams");

        let a_buf = std::mem::replace(&mut stream_buffers[0], MatrixBuffer::dummy(pool));
        let b_buf = std::mem::replace(&mut stream_buffers[1], MatrixBuffer::dummy(pool));
        let batch = a_buf.rows();

        let params = self.store.lock().unwrap().all_params();
        let combiner = Combiner::new(vec![input_dim, input_dim], output_dim);
        let (wa, wb, bias) = combiner.get_weights_and_bias(&params, &slice);

        let a_mat = a_buf.as_mat().to_owned();
        let b_mat = b_buf.as_mat().to_owned();
        let out_mat = combiner.forward_mat(&a_mat, &b_mat, &params, &slice);

        let mut out_buf = pool.acquire(batch, output_dim);
        out_buf.as_mat_mut().copy_from(&out_mat);

        let ctx = DynamicContext::Mat(MatContext::Combiner {
            input_a: a_mat.clone(),
            input_b: b_mat.clone(),
            pre_act: Mat::zeros(batch, output_dim),
        });

        for sample_ctxs in all_ctxs.iter_mut() {
            sample_ctxs.push(ctx.clone());
        }

        pool.release(a_buf);
        pool.release(b_buf);

        *stream_buffers = vec![out_buf];

        let duration = start.elapsed().as_nanos() as f64;
        self.record_segment_timing(seg_index, &device, duration);
    }
}