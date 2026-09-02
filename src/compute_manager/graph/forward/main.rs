// src/compute_manager/graph/forward/main.rs

use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::compute_manager::cpu::parallel::{can_parallelize, forward_universal_parallel};
use crate::compute_manager::dim_change;
use crate::compute_manager::matrix_buffer::{MatrixBufferHandle, TempMatrixPool};
use crate::compute_manager::graph::model::MixedModel;
use crate::compute_manager::graph::types::{ChunkedContexts, Model};
use crate::compute_manager::gpu::processor::process_forward_gpu_buffered;
use crate::device_plan::ComputeDevice;
use crate::layers::{UniversalLayer, UniversalLayerBuffered};

fn get_proc_output_features(
    layers: &[Box<dyn UniversalLayer>],
    input: &MatrixBufferHandle,
) -> usize {
    let mut current_cols = input.cols();
    for layer in layers {
        if let Some(linear) = layer.as_linear() {
            current_cols = <crate::layers::Linear as UniversalLayerBuffered>::output_features(linear);
        } else if layer.as_relu().is_some()
            || layer.as_sigmoid().is_some()
            || layer.as_tanh().is_some()
            || layer.as_leaky_relu().is_some()
            || layer.as_identity().is_some()
            || layer.as_softmax().is_some()
            || layer.as_memory().is_some()
            || layer.as_soft_sparse_gate().is_some()
            || layer.as_soft_keep_gate().is_some()
            || layer.as_dual_anchor().is_some()
        {
            // Размерность не меняется
        }
    }
    current_cols
}

impl MixedModel {
    pub fn forward_mat_multi_buffered(
        &mut self,
        pool: Arc<Mutex<TempMatrixPool>>,
        inputs: Vec<MatrixBufferHandle>,
    ) -> (Vec<MatrixBufferHandle>, ChunkedContexts) {
        assert_eq!(
            inputs.len(),
            self.input_stream_count,
            "forward_mat_multi_buffered: expected {} inputs, got {}",
            self.input_stream_count,
            inputs.len()
        );

        let batch_size = if let Some(first) = inputs.first() {
            first.rows()
        } else {
            return (Vec::new(), Vec::new());
        };
        for buf in &inputs {
            assert_eq!(
                buf.rows(),
                batch_size,
                "All input buffers must have the same number of rows (batch size)"
            );
        }

        let mut stream_buffers: Vec<MatrixBufferHandle> = inputs;
        let mut all_ctxs: ChunkedContexts = Vec::new();

        // Клонируем Arc, а не весь вектор моделей
        let models = Arc::clone(&self.models);

        for (model_index, model) in models.iter().enumerate() {
            let start = Instant::now();
            let device = self.compute_executor.device_for_model(model_index);

            match model {
                Model::Unsqueeze(target_dims) => {
                    let mut new_stream = Vec::with_capacity(stream_buffers.len());
                    for buf in stream_buffers {
                        let mut pool_guard = pool.lock().unwrap();
                        new_stream.push(dim_change::unsqueeze_mat_buffered_handle(&mut pool_guard, buf, target_dims));
                    }
                    stream_buffers = new_stream;
                    self.last_forward_contexts.insert(model_index, Vec::new());
                }
                Model::ReduceMean(target_dims) => {
                    let mut new_stream = Vec::with_capacity(stream_buffers.len());
                    for buf in stream_buffers {
                        let mut pool_guard = pool.lock().unwrap();
                        new_stream.push(dim_change::reduce_mat_buffered_handle(&mut pool_guard, buf, target_dims));
                    }
                    stream_buffers = new_stream;
                    self.last_forward_contexts.insert(model_index, Vec::new());
                }
                Model::UniversalProcessor(proc, slices, stream_indices) => {
                    let active_indices: Vec<usize> = match stream_indices {
                        Some(indices) => indices.clone(),
                        None => (0..stream_buffers.len()).collect(),
                    };

                    let params_handle = self
                        .get_params_handle_for_model(model_index)
                        .unwrap_or_else(|| {
                            let mut pool_guard = pool.lock().unwrap();
                            pool_guard.acquire(0, 0)
                        });

                    for &stream_idx in &active_indices {
                        let input_buf = stream_buffers[stream_idx].clone();

                        if let ComputeDevice::Gpu { .. } = device {
                            let gpu = self.compute_executor.gpu_compute()
                                .expect("GPU requested but not available");

                            let input_gpu = if input_buf.is_gpu() {
                                input_buf
                            } else {
                                let gpu_buf = gpu.allocate_gpu_matrix_handle(
                                    input_buf.rows(),
                                    input_buf.cols(),
                                );
                                gpu.copy_cpu_to_gpu_handle(&input_buf, &gpu_buf);
                                gpu_buf
                            };

                            // Передаём срезы
                            let (out_gpu, layer_ctxs) = process_forward_gpu_buffered(
                                &gpu,
                                proc.as_ref().as_slice(),
                                slices.as_slice(),
                                &params_handle,
                                input_gpu,
                            );

                            stream_buffers[stream_idx] = out_gpu;
                            self.last_forward_contexts
                                .insert(model_index, vec![layer_ctxs.clone()]);
                            all_ctxs.push(layer_ctxs);
                        } else {
                            let can_parallel = can_parallelize(proc.as_ref().as_slice())
                                && input_buf.rows() > 1
                                && self.executor.num_workers() > 1;

                            if can_parallel {
                                let out_features = get_proc_output_features(proc.as_ref().as_slice(), &input_buf);
                                let out_handle = {
                                    let mut pool_guard = pool.lock().unwrap();
                                    pool_guard.acquire(input_buf.rows(), out_features)
                                };

                                // Для параллельной ветки нужны владеющие копии
                                let proc_arc = Arc::clone(proc);
                                let slices_vec = slices.clone();

                                let chunk_ctxs = forward_universal_parallel(
                                    self.executor.as_ref(),
                                    pool.clone(),
                                    proc_arc,
                                    slices_vec,
                                    params_handle.clone(),
                                    input_buf,
                                    out_handle.clone(),
                                );

                                stream_buffers[stream_idx] = out_handle;
                                self.last_forward_contexts
                                    .insert(model_index, chunk_ctxs.clone());
                                all_ctxs.extend(chunk_ctxs);
                            } else {
                                let ctxs = {
                                    let mut pool_guard = pool.lock().unwrap();
                                    self.process_universal_processor_forward_buffered(
                                        &mut pool_guard,
                                        proc,  // передаём &Arc<Vec<...>> как ожидает метод
                                        slices.as_slice(),
                                        model_index,
                                        &params_handle,
                                        &mut stream_buffers,
                                        stream_indices,
                                    )
                                };
                                self.last_forward_contexts
                                    .insert(model_index, vec![ctxs.clone()]);
                                all_ctxs.push(ctxs);
                            }
                        }
                    }
                }
                Model::SplitterConnector { dim_a, dim_b } => {
                    let mut pool_guard = pool.lock().unwrap();
                    self.process_splitter_connector_forward_buffered(
                        &mut pool_guard,
                        *dim_a,
                        *dim_b,
                        batch_size,
                        &mut stream_buffers,
                        &mut all_ctxs,
                        model_index,
                    );
                    self.last_forward_contexts.insert(model_index, Vec::new());
                }
                Model::CombinerConnector { input_dims, .. } => {
                    let mut pool_guard = pool.lock().unwrap();
                    self.process_combiner_connector_forward_buffered(
                        &mut pool_guard,
                        input_dims.clone(),
                        batch_size,
                        &mut stream_buffers,
                        &mut all_ctxs,
                        model_index,
                    );
                    self.last_forward_contexts.insert(model_index, Vec::new());
                }
                Model::Splitter { input_dim, output_dims, slice } => {
                    let mut pool_guard = pool.lock().unwrap();
                    self.process_splitter_forward_buffered(
                        &mut pool_guard,
                        *input_dim,
                        output_dims.clone(),
                        *slice,
                        batch_size,
                        &mut stream_buffers,
                        &mut all_ctxs,
                        model_index,
                    );
                    if let Some(ctx) = all_ctxs.last().and_then(|chunk| chunk.last()) {
                        self.last_forward_contexts
                            .insert(model_index, vec![vec![ctx.clone()]]);
                    } else {
                        self.last_forward_contexts.insert(model_index, Vec::new());
                    }
                }
                Model::Combiner { input_dim, output_dim, slice } => {
                    let mut pool_guard = pool.lock().unwrap();
                    self.process_combiner_forward_buffered(
                        &mut pool_guard,
                        *input_dim,
                        *output_dim,
                        *slice,
                        batch_size,
                        &mut stream_buffers,
                        &mut all_ctxs,
                        model_index,
                    );
                    if let Some(ctx) = all_ctxs.last().and_then(|chunk| chunk.last()) {
                        self.last_forward_contexts
                            .insert(model_index, vec![vec![ctx.clone()]]);
                    } else {
                        self.last_forward_contexts.insert(model_index, Vec::new());
                    }
                }
            }

            let duration = start.elapsed().as_nanos() as f64;
            self.compute_executor
                .record_model_time(model_index, &device, duration);
        }

        assert_eq!(
            stream_buffers.len(),
            self.output_stream_count,
            "forward_mat_multi_buffered: output stream count mismatch"
        );

        (stream_buffers, Vec::new())
    }
}