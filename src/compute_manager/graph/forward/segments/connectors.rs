// src/compute_manager/graph/forward/segments/connectors.rs

use std::time::Instant;

use crate::compute_manager::graph::model::MixedModel;
use crate::compute_manager::graph::types::DynamicContext;
use crate::compute_manager::matrix_buffer::{MatrixBufferHandle, TempMatrixPool};
use crate::device_plan::plan::ComputeDevice;
use crate::layers::splitter::Splitter;
use crate::layers::combiner::Combiner;
use crate::layers::buffered_context::BufferedContext;
use crate::model_plan::param_store::ParamSlice;

impl MixedModel {
    // ===================================================================
    // НОВЫЕ БУФЕРИЗОВАННЫЕ ВЕРСИИ ДЛЯ РАБОТЫ С MatrixBufferHandle
    // ===================================================================

    /// SplitterConnector с управляемыми буферами.
    /// Входные дескрипторы извлекаются, создаются выходные через пул.
    /// Контекст сохраняется как BufferedContext.
    pub(crate) fn process_splitter_connector_forward_buffered(
        &mut self,
        pool: &mut TempMatrixPool,
        _dim_a: usize,
        _dim_b: usize,
        _batch_size: usize,
        stream_buffers: &mut Vec<MatrixBufferHandle>,
        all_ctxs: &mut Vec<Vec<DynamicContext>>,
        seg_index: usize,
    ) {
        let start = Instant::now();
        let device = self.segment_placement
            .get(seg_index)
            .map(|p| p.compute_device.clone())
            .unwrap_or(ComputeDevice::Cpu { id: 0, threads: 1 });

        assert_eq!(stream_buffers.len(), 2, "SplitterConnector buffered: expected 2 input streams");

        // Извлекаем входные дескрипторы
        let input_a = stream_buffers[0].clone();
        let input_b = stream_buffers[1].clone();

        let rows_a = input_a.rows();
        let cols_a = input_a.cols();
        let rows_b = input_b.rows();
        let cols_b = input_b.cols();

        // Создаём выходные буферы того же размера
        let out_a = pool.acquire(rows_a, cols_a);
        let out_b = pool.acquire(rows_b, cols_b);

        // Копируем данные
        copy_handle_data(&input_a, &out_a);
        copy_handle_data(&input_b, &out_b);

        // Сохраняем контекст (Buffered)
        let ctx = DynamicContext::Buffered(BufferedContext::SplitterConnector {
            input: input_a.clone(),
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
    /// Все входные буферы остаются без изменений, только сохраняется контекст.
    pub(crate) fn process_combiner_connector_forward_buffered(
        &mut self,
        _pool: &mut TempMatrixPool,
        input_dims: Vec<usize>,
        _batch_size: usize,
        stream_buffers: &mut Vec<MatrixBufferHandle>,
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

        // Сохраняем контекст со всеми входными дескрипторами
        let inputs = stream_buffers.clone();
        let ctx = DynamicContext::Buffered(BufferedContext::CombinerConnector {
            inputs,
        });
        for sample_ctxs in all_ctxs.iter_mut() {
            sample_ctxs.push(ctx.clone());
        }

        let duration = start.elapsed().as_nanos() as f64;
        self.record_segment_timing(seg_index, &device, duration);
    }

    /// Обучаемый Splitter с управляемыми буферами (CPU-вычисления, GPU-буферы не поддерживаются).
    pub(crate) fn process_splitter_forward_buffered(
        &mut self,
        pool: &mut TempMatrixPool,
        input_dim: usize,
        output_dims: Vec<usize>,
        slice: ParamSlice,
        _batch_size: usize,
        stream_buffers: &mut Vec<MatrixBufferHandle>,
        all_ctxs: &mut Vec<Vec<DynamicContext>>,
        seg_index: usize,
    ) {
        let start = Instant::now();
        let device = self.segment_placement
            .get(seg_index)
            .map(|p| p.compute_device.clone())
            .unwrap_or(ComputeDevice::Cpu { id: 0, threads: 1 });

        assert_eq!(stream_buffers.len(), 1, "Splitter buffered: expected 1 input stream");

        let input_handle = stream_buffers[0].clone();
        let batch = input_handle.rows();
        let params = self.store.lock().unwrap().all_params();
        let splitter = Splitter::new(input_dim, output_dims.clone());
        let (wa_vec, wb_vec, bias_a_vec, bias_b_vec) =
            splitter.get_weights_and_biases_vec(&params, &slice);

        let x_vec = handle_to_vec(&input_handle);

        let p = output_dims[0];
        let q = output_dims[1];

        // Выделяем буферы для выходов и pre-активаций
        let out_a = pool.acquire(batch, p);
        let out_b = pool.acquire(batch, q);
        let pre_a = pool.acquire(batch, p);
        let pre_b = pool.acquire(batch, q);

        // Вычисляем
        {
            let mut out_a_guard = out_a.write();
            let out_a_slice = out_a_guard.as_slice_mut().expect("CPU buffer");
            let mut out_b_guard = out_b.write();
            let out_b_slice = out_b_guard.as_slice_mut().expect("CPU buffer");
            let mut pre_a_guard = pre_a.write();
            let pre_a_slice = pre_a_guard.as_slice_mut().expect("CPU buffer");
            let mut pre_b_guard = pre_b.write();
            let pre_b_slice = pre_b_guard.as_slice_mut().expect("CPU buffer");

            for r in 0..batch {
                for c in 0..p {
                    let mut sum = bias_a_vec[c];
                    for k in 0..input_dim {
                        sum += x_vec[k * batch + r] * wa_vec[c * input_dim + k];
                    }
                    pre_a_slice[c * batch + r] = sum;
                    out_a_slice[c * batch + r] = sum.max(0.0);
                }
                for c in 0..q {
                    let mut sum = bias_b_vec[c];
                    for k in 0..input_dim {
                        sum += x_vec[k * batch + r] * wb_vec[c * input_dim + k];
                    }
                    pre_b_slice[c * batch + r] = sum;
                    out_b_slice[c * batch + r] = sum.max(0.0);
                }
            }
        }

        // Сохраняем контекст
        let ctx = DynamicContext::Buffered(BufferedContext::Splitter {
            input: input_handle.clone(),
            pre_a: pre_a.clone(),
            pre_b: pre_b.clone(),
        });
        for sample_ctxs in all_ctxs.iter_mut() {
            sample_ctxs.push(ctx.clone());
        }

        // Возвращаем входной дескриптор в пул
        pool.release(input_handle);

        *stream_buffers = vec![out_a, out_b];

        let duration = start.elapsed().as_nanos() as f64;
        self.record_segment_timing(seg_index, &device, duration);
    }

    /// Обучаемый Combiner с управляемыми буферами (CPU-вычисления, GPU-буферы не поддерживаются).
    pub(crate) fn process_combiner_forward_buffered(
        &mut self,
        pool: &mut TempMatrixPool,
        input_dim: usize,
        output_dim: usize,
        slice: ParamSlice,
        _batch_size: usize,
        stream_buffers: &mut Vec<MatrixBufferHandle>,
        all_ctxs: &mut Vec<Vec<DynamicContext>>,
        seg_index: usize,
    ) {
        let start = Instant::now();
        let device = self.segment_placement
            .get(seg_index)
            .map(|p| p.compute_device.clone())
            .unwrap_or(ComputeDevice::Cpu { id: 0, threads: 1 });

        assert_eq!(stream_buffers.len(), 2, "Combiner buffered: expected 2 input streams");

        let a_handle = stream_buffers[0].clone();
        let b_handle = stream_buffers[1].clone();

        let batch = a_handle.rows();
        let params = self.store.lock().unwrap().all_params();
        let combiner = Combiner::new(vec![input_dim, input_dim], output_dim);
        let (wa_vec, wb_vec, bias_vec) = combiner.get_weights_and_bias_vec(&params, &slice);

        let a_vec = handle_to_vec(&a_handle);
        let b_vec = handle_to_vec(&b_handle);

        // Выделяем буферы для выхода и pre-активации
        let out_handle = pool.acquire(batch, output_dim);
        let pre_handle = pool.acquire(batch, output_dim);

        // Вычисляем
        {
            let mut out_guard = out_handle.write();
            let out_slice = out_guard.as_slice_mut().expect("CPU buffer");
            let mut pre_guard = pre_handle.write();
            let pre_slice = pre_guard.as_slice_mut().expect("CPU buffer");

            for r in 0..batch {
                for c in 0..output_dim {
                    let mut sum = bias_vec[c];
                    for k in 0..input_dim {
                        sum += a_vec[k * batch + r] * wa_vec[c * input_dim + k];
                        sum += b_vec[k * batch + r] * wb_vec[c * input_dim + k];
                    }
                    pre_slice[c * batch + r] = sum;
                    out_slice[c * batch + r] = sum.max(0.0);
                }
            }
        }

        // Сохраняем контекст
        let ctx = DynamicContext::Buffered(BufferedContext::Combiner {
            input_a: a_handle.clone(),
            input_b: b_handle.clone(),
            pre_act: pre_handle.clone(),
        });
        for sample_ctxs in all_ctxs.iter_mut() {
            sample_ctxs.push(ctx.clone());
        }

        // Возвращаем входные дескрипторы в пул
        pool.release(a_handle);
        pool.release(b_handle);

        *stream_buffers = vec![out_handle];

        let duration = start.elapsed().as_nanos() as f64;
        self.record_segment_timing(seg_index, &device, duration);
    }
}

// Вспомогательные функции для работы с CPU-буферами

/// Копирует данные между двумя дескрипторами (оба должны быть CPU).
fn copy_handle_data(src: &MatrixBufferHandle, dst: &MatrixBufferHandle) {
    let src_guard = src.read();
    let src_slice = src_guard.as_slice().expect("Source must be CPU");
    let mut dst_guard = dst.write();
    let dst_slice = dst_guard.as_slice_mut().expect("Destination must be CPU");
    assert_eq!(src_slice.len(), dst_slice.len());
    dst_slice.copy_from_slice(src_slice);
}

/// Читает данные из CPU handle в Vec<f32> (column-major порядок).
fn handle_to_vec(handle: &MatrixBufferHandle) -> Vec<f32> {
    assert!(!handle.is_gpu(), "handle_to_vec supports only CPU buffers");
    let guard = handle.read();
    guard.as_slice().expect("CPU buffer").to_vec()
}