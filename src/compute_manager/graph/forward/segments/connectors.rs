// src/compute_manager/graph/forward/segments/connectors.rs

use std::time::Instant;

use crate::compute_manager::graph::model::MixedModel;
use crate::compute_manager::graph::types::DynamicContext;
use crate::compute_manager::matrix_buffer::{MatrixBufferHandle, TempMatrixPool};
use crate::device_plan::plan::ComputeDevice;
use crate::layers::buffered_context::BufferedContext;
use crate::model_plan::param_store::ParamSlice;

impl MixedModel {
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
        let device = self.compute_executor.device_for_segment(seg_index);

        assert_eq!(stream_buffers.len(), 2, "SplitterConnector buffered: expected 2 input streams");

        let input_a = stream_buffers[0].clone();
        let input_b = stream_buffers[1].clone();

        let rows_a = input_a.rows();
        let cols_a = input_a.cols();
        let rows_b = input_b.rows();
        let cols_b = input_b.cols();

        let out_a = pool.acquire(rows_a, cols_a);
        let out_b = pool.acquire(rows_b, cols_b);

        {
            let mut mem = self.memory_executor.lock().unwrap();
            mem.copy_cpu_buffer(input_a.id(), out_a.id());
            mem.copy_cpu_buffer(input_b.id(), out_b.id());
        }

        let ctx = DynamicContext::Buffered(BufferedContext::SplitterConnector {
            input: input_a.clone(),
        });
        for sample_ctxs in all_ctxs.iter_mut() {
            sample_ctxs.push(ctx.clone());
        }

        pool.release(input_a);
        pool.release(input_b);

        *stream_buffers = vec![out_a, out_b];

        let duration = start.elapsed().as_nanos() as f64;
        self.compute_executor.record_segment_time(seg_index, &device, duration);
    }

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
        let device = self.compute_executor.device_for_segment(seg_index);

        let n = input_dims.len();
        assert_eq!(stream_buffers.len(), n,
            "CombinerConnector buffered: expected {} input streams, got {}",
            n, stream_buffers.len());

        let inputs = stream_buffers.clone();
        let ctx = DynamicContext::Buffered(BufferedContext::CombinerConnector {
            inputs,
        });
        for sample_ctxs in all_ctxs.iter_mut() {
            sample_ctxs.push(ctx.clone());
        }

        let duration = start.elapsed().as_nanos() as f64;
        self.compute_executor.record_segment_time(seg_index, &device, duration);
    }

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
        let device = self.compute_executor.device_for_segment(seg_index);

        assert_eq!(stream_buffers.len(), 1, "Splitter buffered: expected 1 input stream");

        let input_handle = stream_buffers[0].clone();
        let batch = input_handle.rows();

        // Получаем параметры сегмента через ParamStore, используя слайс.
        let params_handle = {
            let ps = self.param_store.lock().unwrap();
            ps.params_handle(&slice).clone()
        };

        let p = output_dims[0];
        let q = output_dims[1];

        let out_a = pool.acquire(batch, p);
        let out_b = pool.acquire(batch, q);
        let pre_a = pool.acquire(batch, p);
        let pre_b = pool.acquire(batch, q);

        {
            let ids = [
                input_handle.id(),
                out_a.id(),
                pre_a.id(),
                out_b.id(),
                pre_b.id(),
                params_handle.id(),
            ];
            input_handle.memory().lock().unwrap().with_cpu_slices_mut(&ids, |slices| {
                let (first, rest) = slices.split_at_mut(1);
                let x: &[f32] = &*first[0];
                let (second, rest) = rest.split_at_mut(1);
                let a_out: &mut [f32] = &mut *second[0];
                let (third, rest) = rest.split_at_mut(1);
                let a_pre: &mut [f32] = &mut *third[0];
                let (fourth, rest) = rest.split_at_mut(1);
                let b_out: &mut [f32] = &mut *fourth[0];
                let (fifth, rest) = rest.split_at_mut(1);
                let b_pre: &mut [f32] = &mut *fifth[0];
                let params: &[f32] = &*rest[0];

                let wa_start = slice.start;
                let wa_len = p * input_dim;
                let wb_start = wa_start + wa_len;
                let wb_len = q * input_dim;
                let bias_a_start = wb_start + wb_len;
                let bias_b_start = bias_a_start + p;

                for r in 0..batch {
                    for c in 0..p {
                        let mut sum = params[bias_a_start + c];
                        for k in 0..input_dim {
                            sum += x[k * batch + r] * params[wa_start + c * input_dim + k];
                        }
                        a_pre[c * batch + r] = sum;
                        a_out[c * batch + r] = sum.max(0.0);
                    }
                    for c in 0..q {
                        let mut sum = params[bias_b_start + c];
                        for k in 0..input_dim {
                            sum += x[k * batch + r] * params[wb_start + c * input_dim + k];
                        }
                        b_pre[c * batch + r] = sum;
                        b_out[c * batch + r] = sum.max(0.0);
                    }
                }
            });
        }

        let ctx = DynamicContext::Buffered(BufferedContext::Splitter {
            input: input_handle.clone(),
            pre_a: pre_a.clone(),
            pre_b: pre_b.clone(),
        });
        for sample_ctxs in all_ctxs.iter_mut() {
            sample_ctxs.push(ctx.clone());
        }

        pool.release(input_handle);

        *stream_buffers = vec![out_a, out_b];

        let duration = start.elapsed().as_nanos() as f64;
        self.compute_executor.record_segment_time(seg_index, &device, duration);
    }

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
        let device = self.compute_executor.device_for_segment(seg_index);

        assert_eq!(stream_buffers.len(), 2, "Combiner buffered: expected 2 input streams");

        let a_handle = stream_buffers[0].clone();
        let b_handle = stream_buffers[1].clone();
        let batch = a_handle.rows();

        // Получаем параметры сегмента через ParamStore, используя слайс.
        let params_handle = {
            let ps = self.param_store.lock().unwrap();
            ps.params_handle(&slice).clone()
        };

        let out_handle = pool.acquire(batch, output_dim);
        let pre_handle = pool.acquire(batch, output_dim);

        {
            let ids = [
                a_handle.id(),
                b_handle.id(),
                out_handle.id(),
                pre_handle.id(),
                params_handle.id(),
            ];
            a_handle.memory().lock().unwrap().with_cpu_slices_mut(&ids, |slices| {
                let (first, rest) = slices.split_at_mut(1);
                let a: &[f32] = &*first[0];
                let (second, rest) = rest.split_at_mut(1);
                let b: &[f32] = &*second[0];
                let (third, rest) = rest.split_at_mut(1);
                let out_val: &mut [f32] = &mut *third[0];
                let (fourth, rest) = rest.split_at_mut(1);
                let pre_val: &mut [f32] = &mut *fourth[0];
                let params: &[f32] = &*rest[0];

                let wa_start = slice.start;
                let wa_len = output_dim * input_dim;
                let wb_start = wa_start + wa_len;
                let wb_len = output_dim * input_dim;
                let bias_start = wb_start + wb_len;

                for r in 0..batch {
                    for c in 0..output_dim {
                        let mut sum = params[bias_start + c];
                        for k in 0..input_dim {
                            sum += a[k * batch + r] * params[wa_start + c * input_dim + k];
                            sum += b[k * batch + r] * params[wb_start + c * input_dim + k];
                        }
                        pre_val[c * batch + r] = sum;
                        out_val[c * batch + r] = sum.max(0.0);
                    }
                }
            });
        }

        let ctx = DynamicContext::Buffered(BufferedContext::Combiner {
            input_a: a_handle.clone(),
            input_b: b_handle.clone(),
            pre_act: pre_handle.clone(),
        });
        for sample_ctxs in all_ctxs.iter_mut() {
            sample_ctxs.push(ctx.clone());
        }

        pool.release(a_handle);
        pool.release(b_handle);

        *stream_buffers = vec![out_handle];

        let duration = start.elapsed().as_nanos() as f64;
        self.compute_executor.record_segment_time(seg_index, &device, duration);
    }
}