// src/compute_manager/graph/backward/main.rs

use std::sync::Arc;
use std::time::Instant;

use crate::compute_manager::cpu::parallel::{can_parallelize, backward_universal_parallel};
use crate::compute_manager::dim_change;
use crate::compute_manager::matrix_buffer::{MatrixBufferHandle, TempMatrixPool};
use crate::compute_manager::graph::model::MixedModel;
use crate::compute_manager::graph::types::{DynamicContext, Model};
use crate::compute_manager::gpu::processor::process_backward_gpu_buffered;
use crate::device_plan::ComputeDevice;
use crate::layers::{UniversalLayer, UniversalLayerBuffered};
use crate::model_plan::param_store::ParamSlice;

use super::segments::{processors::*, connectors::*};

impl MixedModel {
    pub fn backward_mat_multi_buffered(
        &mut self,
        deltas: Vec<MatrixBufferHandle>,
    ) -> Vec<MatrixBufferHandle> {
        assert_eq!(deltas.len(), self.output_stream_count,
            "backward_mat_multi_buffered: expected {} deltas, got {}",
            self.output_stream_count, deltas.len());

        let mut stream_gradients = deltas;
        let models = Arc::clone(&self.models);

        let pool = self.temp_matrix_pool.clone();

        for (model_index, model) in models.iter().enumerate().rev() {
            let start = Instant::now();
            let device = self.compute_executor.device_for_model(model_index);

            let params_handle = self
                .get_params_handle_for_model(model_index)
                .unwrap_or_else(|| {
                    let mut pool_guard = pool.lock().unwrap();
                    pool_guard.acquire(0, 0)
                });
            let grad_params_handle = self
                .get_grads_handle_for_model(model_index)
                .unwrap_or_else(|| {
                    let mut pool_guard = pool.lock().unwrap();
                    pool_guard.acquire(0, 0)
                });

            let chunked_ctxs = self
                .last_forward_contexts
                .get(&model_index)
                .cloned()
                .unwrap_or_default();

            match model {
                Model::Unsqueeze(target_dims) => {
                    let mut new_stream = Vec::with_capacity(stream_gradients.len());
                    for buf in stream_gradients {
                        let mut pool_guard = pool.lock().unwrap();
                        new_stream.push(dim_change::reduce_mat_buffered_handle(&mut pool_guard, buf, target_dims));
                    }
                    stream_gradients = new_stream;
                }
                Model::ReduceMean(target_dims) => {
                    let mut new_stream = Vec::with_capacity(stream_gradients.len());
                    for buf in stream_gradients {
                        let mut pool_guard = pool.lock().unwrap();
                        new_stream.push(dim_change::unsqueeze_mat_buffered_handle(&mut pool_guard, buf, target_dims));
                    }
                    stream_gradients = new_stream;
                }
                Model::UniversalProcessor(proc, slices, stream_indices) => {
                    let active_indices: Vec<usize> = match stream_indices {
                        Some(indices) => indices.clone(),
                        None => (0..stream_gradients.len()).collect(),
                    };

                    let mut new_gradients: Vec<Option<MatrixBufferHandle>> =
                        (0..stream_gradients.len()).map(|_| None).collect();

                    for &stream_idx in &active_indices {
                        let delta_handle = stream_gradients[stream_idx].clone();

                        let can_parallel = matches!(device, ComputeDevice::Cpu { .. })
                            && can_parallelize(proc.as_ref().as_slice())
                            && delta_handle.rows() > 1
                            && self.executor.num_workers() > 1
                            && chunked_ctxs.len() > 1;

                        if can_parallel {
                            let batch = delta_handle.rows();
                            let input_features = if let Some(linear) = proc.first().and_then(|l| l.as_linear()) {
                                <dyn UniversalLayerBuffered>::input_features(linear)
                            } else {
                                delta_handle.cols()
                            };

                            let grad_input_handle = {
                                let mut pool_guard = pool.lock().unwrap();
                                pool_guard.acquire(batch, input_features)
                            };

                            let proc_arc = Arc::clone(proc);
                            let slices_vec = slices.clone();
                            let ctx_chunk = chunked_ctxs.clone();

                            backward_universal_parallel(
                                self.executor.as_ref(),
                                pool.clone(),
                                proc_arc,
                                slices_vec,
                                ctx_chunk,
                                delta_handle,
                                grad_input_handle.clone(),
                                params_handle.clone(),
                                grad_params_handle.clone(),
                            );

                            new_gradients[stream_idx] = Some(grad_input_handle);
                        } else if let ComputeDevice::Gpu { .. } = device {
                            let gpu = self.compute_executor.gpu_compute()
                                .expect("GPU requested but not available");

                            let delta_gpu_handle = if delta_handle.is_gpu() {
                                delta_handle
                            } else {
                                let gpu_handle = gpu.allocate_gpu_matrix_handle(
                                    delta_handle.rows(),
                                    delta_handle.cols(),
                                );
                                gpu.copy_cpu_to_gpu_handle(&delta_handle, &gpu_handle);
                                gpu_handle
                            };

                            let layer_ctxs = if chunked_ctxs.is_empty() {
                                Vec::new()
                            } else {
                                chunked_ctxs[0].clone()
                            };
                            let ctxs_owned: Vec<DynamicContext> = layer_ctxs;
                            let ctxs_slice: &[DynamicContext] = &ctxs_owned;

                            let out_gpu = process_backward_gpu_buffered(
                                &gpu,
                                proc.as_ref().as_slice(),
                                slices.as_slice(),
                                ctxs_slice,
                                &params_handle,
                                delta_gpu_handle,
                                &grad_params_handle,
                            );

                            let cpu_handle = {
                                let mut pool_guard = pool.lock().unwrap();
                                let handle = pool_guard.acquire(out_gpu.rows(), out_gpu.cols());
                                gpu.copy_gpu_to_cpu_handle(&out_gpu, &handle);
                                handle
                            };
                            new_gradients[stream_idx] = Some(cpu_handle);
                        } else {
                            let layer_ctxs = if chunked_ctxs.is_empty() {
                                Vec::new()
                            } else {
                                chunked_ctxs[0].clone()
                            };
                            let ctxs_refs: Vec<&DynamicContext> = layer_ctxs.iter().collect();

                            let in_delta_handle = {
                                let mut pool_guard = pool.lock().unwrap();
                                self.backward_universal_batch_buffered_handle(
                                    &mut pool_guard,
                                    proc.as_ref().as_slice(),
                                    slices.as_slice(),
                                    &ctxs_refs,
                                    delta_handle,
                                    &params_handle,
                                    &grad_params_handle,
                                )
                            };
                            new_gradients[stream_idx] = Some(in_delta_handle);
                        }
                    }

                    let mut final_grads = Vec::with_capacity(stream_gradients.len());
                    for i in 0..stream_gradients.len() {
                        if let Some(handle) = new_gradients[i].take() {
                            final_grads.push(handle);
                        } else {
                            final_grads.push(stream_gradients[i].clone());
                        }
                    }
                    stream_gradients = final_grads;
                }
                Model::SplitterConnector { .. } => {
                    let mut pool_guard = pool.lock().unwrap();
                    stream_gradients = self.process_splitter_connector_backward_buffered(
                        &mut pool_guard,
                        stream_gradients,
                    );
                }
                Model::CombinerConnector { .. } => {
                    // Ничего не делаем, градиенты остаются без изменений
                }
                Model::Splitter { input_dim, output_dims, slice } => {
                    let mut pool_guard = pool.lock().unwrap();
                    stream_gradients = self.process_splitter_backward_buffered(
                        &mut pool_guard,
                        *input_dim,
                        output_dims,
                        *slice,
                        &chunked_ctxs,
                        &params_handle,
                        &grad_params_handle,
                        stream_gradients,
                    );
                }
                Model::Combiner { input_dim, output_dim, slice } => {
                    let mut pool_guard = pool.lock().unwrap();
                    stream_gradients = self.process_combiner_backward_buffered(
                        &mut pool_guard,
                        *input_dim,
                        *output_dim,
                        *slice,
                        &chunked_ctxs,
                        &params_handle,
                        &grad_params_handle,
                        stream_gradients,
                    );
                }
            }

            let duration = start.elapsed().as_nanos() as f64;
            self.compute_executor.record_model_time(model_index, &device, duration);
        }

        assert_eq!(stream_gradients.len(), self.input_stream_count);
        stream_gradients
    }
}