// src/compute_manager/graph/backward/segments.rs

use std::time::Instant;

use crate::compute_manager::dim_change;
use crate::compute_manager::graph::model::MixedModel;
use crate::compute_manager::matrix_buffer::{MatrixBufferHandle, TempMatrixPool};
use crate::device_plan::plan::ComputeDevice;

impl MixedModel {
    // ===================================================================
    // Handle-версии (MatrixBufferHandle + TempMatrixPool)
    // ===================================================================

    /// Обратный проход для операции Unsqueeze (handle-версия).
    /// Выполняет уменьшение размерности (reduce) над всеми потоковыми дескрипторами.
    /// Старые входные дескрипторы возвращаются в пул, новые берутся из пула.
    pub(super) fn process_unsqueeze_backward_buffered_handle(
        &mut self,
        pool: &mut TempMatrixPool,
        stream_buffers: &mut Vec<MatrixBufferHandle>,
        target_dims: &[usize],
        seg_index: usize,
    ) {
        let start = Instant::now();

        let device = self
            .segment_placement
            .get(seg_index)
            .map(|p| p.compute_device.clone())
            .unwrap_or(ComputeDevice::Cpu { id: 0, threads: 1 });

        // Перебираем все потоковые буферы и преобразуем их
        let mut new_stream = Vec::with_capacity(stream_buffers.len());
        for buf in stream_buffers.drain(..) {
            // Выполняем reduce над handle
            let new_buf = dim_change::reduce_mat_buffered_handle(pool, buf, target_dims);
            new_stream.push(new_buf);
        }
        *stream_buffers = new_stream;

        let duration = start.elapsed().as_nanos() as f64;
        self.record_segment_timing(seg_index, &device, duration);
    }

    /// Обратный проход для операции ReduceMean (handle-версия).
    /// Выполняет увеличение размерности (unsqueeze) над всеми потоковыми дескрипторами.
    pub(super) fn process_reduce_mean_backward_buffered_handle(
        &mut self,
        pool: &mut TempMatrixPool,
        stream_buffers: &mut Vec<MatrixBufferHandle>,
        target_dims: &[usize],
        seg_index: usize,
    ) {
        let start = Instant::now();

        let device = self
            .segment_placement
            .get(seg_index)
            .map(|p| p.compute_device.clone())
            .unwrap_or(ComputeDevice::Cpu { id: 0, threads: 1 });

        let mut new_stream = Vec::with_capacity(stream_buffers.len());
        for buf in stream_buffers.drain(..) {
            let new_buf = dim_change::unsqueeze_mat_buffered_handle(pool, buf, target_dims);
            new_stream.push(new_buf);
        }
        *stream_buffers = new_stream;

        let duration = start.elapsed().as_nanos() as f64;
        self.record_segment_timing(seg_index, &device, duration);
    }
}