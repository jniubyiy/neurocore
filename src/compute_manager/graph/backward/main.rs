// src/compute_manager/graph/backward/main.rs

use std::time::Instant;

use crate::compute_manager::dim_change;
use crate::compute_manager::matrix_buffer::{MatrixBufferHandle, TempMatrixPool};
use crate::compute_manager::graph::model::MixedModel;
use crate::compute_manager::graph::types::{DynamicContext, Segment};
use crate::compute_manager::gpu::processor::process_backward_gpu_buffered;
use crate::device_plan::plan::ComputeDevice;
use crate::layers::{
    UniversalLayer, UniversalLayerBuffered,
    Linear, ReLU, Sigmoid, Tanh, LeakyReLU, Identity, Softmax,
    Memory, SoftSparseGate, SoftKeepGate, DualAnchor,
};
use crate::model_plan::param_store::ParamSlice;

impl MixedModel {
    // ===================================================================
    // Буферизованная версия обратного прохода (MatrixBufferHandle + TempMatrixPool)
    // ===================================================================

    pub fn backward_mat_multi_buffered(
        &mut self,
        pool: &mut TempMatrixPool,
        contexts: &[Vec<DynamicContext>],
        deltas: Vec<MatrixBufferHandle>,
    ) -> Vec<MatrixBufferHandle> {
        assert_eq!(deltas.len(), self.output_stream_count,
            "backward_mat_multi_buffered: expected {} deltas, got {}",
            self.output_stream_count, deltas.len());

        // Получаем параметры и глобальный буфер градиентов
        let params = self.buffered_param_store.lock().unwrap().get_all_params();
        let grad_params_handle = self.buffered_param_store.lock().unwrap().grads_handle().clone();

        let mut stream_gradients = deltas;
        let total_context_len = contexts.first().map(|c| c.len()).unwrap_or(0);
        let mut ctx_pos = total_context_len;

        let segments = self.segments.clone();

        for (seg_index, seg) in segments.iter().enumerate().rev() {
            let start = Instant::now();

            match seg {
                Segment::Unsqueeze(target_dims) => {
                    let mut new_stream = Vec::with_capacity(stream_gradients.len());
                    for buf in stream_gradients {
                        new_stream.push(dim_change::reduce_mat_buffered_handle(pool, buf, target_dims));
                    }
                    stream_gradients = new_stream;
                }
                Segment::ReduceMean(target_dims) => {
                    let mut new_stream = Vec::with_capacity(stream_gradients.len());
                    for buf in stream_gradients {
                        new_stream.push(dim_change::unsqueeze_mat_buffered_handle(pool, buf, target_dims));
                    }
                    stream_gradients = new_stream;
                }
                Segment::UniversalProcessor(proc, slices, stream_indices) => {
                    let num_layers = proc.len();
                    let active_indices: Vec<usize> = match stream_indices {
                        Some(indices) => indices.clone(),
                        None => (0..stream_gradients.len()).collect(),
                    };

                    let mut new_gradients: Vec<Option<MatrixBufferHandle>> =
                        (0..stream_gradients.len()).map(|_| None).collect();

                    for &stream_idx in &active_indices {
                        let delta_handle = stream_gradients[stream_idx].clone();
                        let pos_in_sorted = active_indices.iter().position(|&x| x == stream_idx).unwrap();
                        let stream_ctx_start = ctx_pos - (active_indices.len() - pos_in_sorted) * num_layers;
                        let layer_ctxs: Vec<&DynamicContext> = contexts[0]
                            [stream_ctx_start..stream_ctx_start + num_layers]
                            .iter()
                            .collect();

                        let ctxs_owned: Vec<DynamicContext> = layer_ctxs.iter().map(|&c| c.clone()).collect();
                        let ctxs_slice: &[DynamicContext] = &ctxs_owned;

                        if self.gpu_compute.is_some() {
                            // GPU-ветка
                            let gpu = self.gpu_compute.as_ref().unwrap().lock().unwrap();

                            let delta_gpu_handle = if delta_handle.is_gpu() {
                                delta_handle.clone()
                            } else {
                                let gpu_handle = gpu.allocate_gpu_matrix_handle(
                                    delta_handle.rows(),
                                    delta_handle.cols(),
                                );
                                gpu.copy_cpu_to_gpu_handle(&delta_handle, &gpu_handle);
                                gpu_handle
                            };

                            let out_gpu = process_backward_gpu_buffered(
                                &gpu,
                                proc,
                                slices,
                                ctxs_slice,
                                &params,
                                delta_gpu_handle,
                                &grad_params_handle,
                            );

                            let out_mat = gpu.download_gpu_handle_to_mat(&out_gpu);
                            let cpu_handle = pool.acquire(out_mat.nrows(), out_mat.ncols());
                            {
                                let mut guard = cpu_handle.write();
                                let dst = guard.as_slice_mut().expect("CPU buffer");
                                for c in 0..out_mat.ncols() {
                                    for r in 0..out_mat.nrows() {
                                        dst[c * out_mat.nrows() + r] = out_mat[(r, c)];
                                    }
                                }
                            }
                            new_gradients[stream_idx] = Some(cpu_handle);
                        } else {
                            // CPU-путь
                            let in_delta_handle = self.backward_universal_batch_buffered_handle(
                                pool,
                                proc,
                                slices,
                                &layer_ctxs,
                                delta_handle,
                                &params,
                                &grad_params_handle,
                            );
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

                    ctx_pos -= num_layers * active_indices.len();
                }
                Segment::SplitterConnector { .. } => {
                    assert_eq!(stream_gradients.len(), 2);
                    let delta_a = stream_gradients[0].clone();
                    let delta_b = stream_gradients[1].clone();

                    let in_a = pool.acquire(delta_a.rows(), delta_a.cols());
                    let in_b = pool.acquire(delta_b.rows(), delta_b.cols());
                    copy_handle_data(&delta_a, &in_a);
                    copy_handle_data(&delta_b, &in_b);

                    pool.release(delta_a);
                    pool.release(delta_b);

                    stream_gradients = vec![in_a, in_b];
                    ctx_pos -= 1;
                }
                Segment::CombinerConnector { .. } => {
                    ctx_pos -= 1;
                }
                Segment::Splitter {
                    input_dim,
                    output_dims,
                    slice,
                } => {
                    assert!(ctx_pos > 0);
                    let ctx = &contexts[0][ctx_pos - 1];
                    let (x_handle, pre_a_handle, pre_b_handle) = match ctx {
                        DynamicContext::Buffered(crate::layers::buffered_context::BufferedContext::Splitter {
                            input,
                            pre_a,
                            pre_b,
                        }) => (input.clone(), pre_a.clone(), pre_b.clone()),
                        _ => panic!("Expected Splitter Buffered context"),
                    };

                    let da_handle = stream_gradients[0].clone();
                    let db_handle = stream_gradients[1].clone();

                    let x_vec = handle_to_vec(&x_handle);
                    let da_vec = handle_to_vec(&da_handle);
                    let db_vec = handle_to_vec(&db_handle);
                    let pre_a_vec = handle_to_vec(&pre_a_handle);
                    let pre_b_vec = handle_to_vec(&pre_b_handle);

                    let splitter = crate::layers::Splitter::new(*input_dim, output_dims.clone());
                    let (wa_vec, wb_vec, _, _) = splitter.get_weights_and_biases_vec(&params, slice);

                    let batch = x_handle.rows();
                    let n = *input_dim;
                    let p = output_dims[0];
                    let q = output_dims[1];

                    let mut d_pre_a_vec = vec![0.0f32; batch * p];
                    let mut d_pre_b_vec = vec![0.0f32; batch * q];
                    for i in 0..batch * p {
                        d_pre_a_vec[i] = if pre_a_vec[i] > 0.0 { da_vec[i] } else { 0.0 };
                    }
                    for i in 0..batch * q {
                        d_pre_b_vec[i] = if pre_b_vec[i] > 0.0 { db_vec[i] } else { 0.0 };
                    }

                    let mut dx_vec = vec![0.0f32; batch * n];
                    for r in 0..batch {
                        for c in 0..n {
                            let mut sum = 0.0;
                            for k in 0..p {
                                sum += d_pre_a_vec[k * batch + r] * wa_vec[k * n + c];
                            }
                            for k in 0..q {
                                sum += d_pre_b_vec[k * batch + r] * wb_vec[k * n + c];
                            }
                            dx_vec[c * batch + r] = sum;
                        }
                    }

                    // Записываем градиенты параметров прямо в глобальный буфер
                    grad_params_handle.with_cpu_data_mut(|grad_data| {
                        // d_wa
                        for out_idx in 0..p {
                            for in_idx in 0..n {
                                let mut sum = 0.0;
                                for r in 0..batch {
                                    sum += d_pre_a_vec[out_idx * batch + r] * x_vec[in_idx * batch + r];
                                }
                                grad_data[slice.start + out_idx * n + in_idx] = sum;
                            }
                        }

                        // d_wb
                        let offset = slice.start + p * n;
                        for out_idx in 0..q {
                            for in_idx in 0..n {
                                let mut sum = 0.0;
                                for r in 0..batch {
                                    sum += d_pre_b_vec[out_idx * batch + r] * x_vec[in_idx * batch + r];
                                }
                                grad_data[offset + out_idx * n + in_idx] = sum;
                            }
                        }

                        // d_bias_a
                        let offset = offset + q * n;
                        for c in 0..p {
                            let mut sum = 0.0;
                            for r in 0..batch {
                                sum += d_pre_a_vec[c * batch + r];
                            }
                            grad_data[offset + c] = sum;
                        }

                        // d_bias_b
                        let offset = offset + p;
                        for c in 0..q {
                            let mut sum = 0.0;
                            for r in 0..batch {
                                sum += d_pre_b_vec[c * batch + r];
                            }
                            grad_data[offset + c] = sum;
                        }
                    });

                    let mut dx_handle = pool.acquire(batch, n);
                    vec_to_handle(&dx_vec, &mut dx_handle);

                    pool.release(da_handle);
                    pool.release(db_handle);
                    pool.release(x_handle);
                    pool.release(pre_a_handle);
                    pool.release(pre_b_handle);

                    stream_gradients = vec![dx_handle];
                    ctx_pos -= 1;
                }
                Segment::Combiner {
                    input_dim,
                    output_dim,
                    slice,
                } => {
                    assert!(ctx_pos > 0);
                    let ctx = &contexts[0][ctx_pos - 1];
                    let (a_handle, b_handle, pre_handle) = match ctx {
                        DynamicContext::Buffered(crate::layers::buffered_context::BufferedContext::Combiner {
                            input_a,
                            input_b,
                            pre_act,
                        }) => (input_a.clone(), input_b.clone(), pre_act.clone()),
                        _ => panic!("Expected Combiner Buffered context"),
                    };

                    let dout_handle = stream_gradients[0].clone();

                    let a_vec = handle_to_vec(&a_handle);
                    let b_vec = handle_to_vec(&b_handle);
                    let pre_vec = handle_to_vec(&pre_handle);
                    let dout_vec = handle_to_vec(&dout_handle);

                    let combiner = crate::layers::Combiner::new(vec![*input_dim, *input_dim], *output_dim);
                    let (wa_vec, wb_vec, _) = combiner.get_weights_and_bias_vec(&params, slice);

                    let batch = a_handle.rows();
                    let n = *input_dim;
                    let m = *output_dim;

                    let mut d_pre_vec = vec![0.0f32; batch * m];
                    for i in 0..batch * m {
                        d_pre_vec[i] = if pre_vec[i] > 0.0 { dout_vec[i] } else { 0.0 };
                    }

                    let mut da_vec = vec![0.0f32; batch * n];
                    let mut db_vec = vec![0.0f32; batch * n];
                    for r in 0..batch {
                        for c in 0..n {
                            let mut sum_a = 0.0;
                            let mut sum_b = 0.0;
                            for k in 0..m {
                                sum_a += d_pre_vec[k * batch + r] * wa_vec[k * n + c];
                                sum_b += d_pre_vec[k * batch + r] * wb_vec[k * n + c];
                            }
                            da_vec[c * batch + r] = sum_a;
                            db_vec[c * batch + r] = sum_b;
                        }
                    }

                    // Записываем градиенты параметров прямо в глобальный буфер
                    grad_params_handle.with_cpu_data_mut(|grad_data| {
                        // d_wa
                        for out_idx in 0..m {
                            for in_idx in 0..n {
                                let mut sum = 0.0;
                                for r in 0..batch {
                                    sum += d_pre_vec[out_idx * batch + r] * a_vec[in_idx * batch + r];
                                }
                                grad_data[slice.start + out_idx * n + in_idx] = sum;
                            }
                        }

                        // d_wb
                        let offset = slice.start + m * n;
                        for out_idx in 0..m {
                            for in_idx in 0..n {
                                let mut sum = 0.0;
                                for r in 0..batch {
                                    sum += d_pre_vec[out_idx * batch + r] * b_vec[in_idx * batch + r];
                                }
                                grad_data[offset + out_idx * n + in_idx] = sum;
                            }
                        }

                        // d_bias
                        let offset = offset + m * n;
                        for c in 0..m {
                            let mut sum = 0.0;
                            for r in 0..batch {
                                sum += d_pre_vec[c * batch + r];
                            }
                            grad_data[offset + c] = sum;
                        }
                    });

                    let mut da_handle = pool.acquire(batch, n);
                    let mut db_handle = pool.acquire(batch, n);
                    vec_to_handle(&da_vec, &mut da_handle);
                    vec_to_handle(&db_vec, &mut db_handle);

                    pool.release(dout_handle);
                    pool.release(a_handle);
                    pool.release(b_handle);
                    pool.release(pre_handle);

                    stream_gradients = vec![da_handle, db_handle];
                    ctx_pos -= 1;
                }
            }

            let duration = start.elapsed().as_nanos() as f64;
            let device = self.segment_placement
                .get(seg_index)
                .map(|p| p.compute_device.clone())
                .unwrap_or(ComputeDevice::Cpu { id: 0, threads: 1 });
            self.record_segment_timing(seg_index, &device, duration);
        }

        assert_eq!(stream_gradients.len(), self.input_stream_count);
        stream_gradients
    }

    /// Универсальный обратный проход через слои с использованием MatrixBufferHandle
    fn backward_universal_batch_buffered_handle(
        &mut self,
        pool: &mut TempMatrixPool,
        layers: &[Box<dyn UniversalLayer>],
        slices: &[ParamSlice],
        ctxs: &[&DynamicContext],
        grad_out: MatrixBufferHandle,
        params: &[f32],
        grad_params_handle: &MatrixBufferHandle,
    ) -> MatrixBufferHandle {
        let mut current_grad = grad_out;
        for i in (0..layers.len()).rev() {
            let layer = &layers[i];
            let slice = &slices[i];
            let ctx = ctxs[i];

            let in_features = if let Some(l) = layer.as_linear() {
                <dyn UniversalLayerBuffered>::input_features(l)
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
                current_grad.cols()
            } else {
                current_grad.cols()
            };

            let batch = current_grad.rows();
            let mut grad_input = pool.acquire(batch, in_features);

            call_backward_buffered(
                layer,
                ctx,
                &current_grad,
                &mut grad_input,
                params,
                slice,
                grad_params_handle,
            );

            pool.release(current_grad);
            current_grad = grad_input;
        }
        current_grad
    }
}

// Вспомогательные функции для работы с CPU-буферами

/// Читает данные из CPU handle в Vec<f32> (column-major порядок).
fn handle_to_vec(handle: &MatrixBufferHandle) -> Vec<f32> {
    assert!(!handle.is_gpu(), "handle_to_vec supports only CPU buffers");
    let guard = handle.read();
    guard.as_slice().expect("CPU buffer").to_vec()
}

/// Записывает Vec<f32> в CPU handle (column-major порядок).
fn vec_to_handle(data: &[f32], handle: &mut MatrixBufferHandle) {
    assert!(!handle.is_gpu(), "vec_to_handle supports only CPU buffers");
    let mut guard = handle.write();
    let dst = guard.as_slice_mut().expect("CPU buffer");
    assert_eq!(data.len(), dst.len());
    dst.copy_from_slice(data);
}

/// Копирует данные между двумя CPU handle.
fn copy_handle_data(src: &MatrixBufferHandle, dst: &MatrixBufferHandle) {
    let src_guard = src.read();
    let src_slice = src_guard.as_slice().expect("copy_handle_data: source must be CPU");
    let mut dst_guard = dst.write();
    let dst_slice = dst_guard.as_slice_mut().expect("copy_handle_data: destination must be CPU");
    assert_eq!(src_slice.len(), dst_slice.len());
    dst_slice.copy_from_slice(src_slice);
}

/// Вызывает `backward_buffered` для конкретного слоя с учётом новой сигнатуры.
fn call_backward_buffered(
    layer: &Box<dyn UniversalLayer>,
    ctx: &DynamicContext,
    grad_output: &MatrixBufferHandle,
    grad_input: &mut MatrixBufferHandle,
    params: &[f32],
    slice: &ParamSlice,
    grad_params_handle: &MatrixBufferHandle,
) {
    if let Some(linear) = layer.as_linear() {
        <Linear as UniversalLayerBuffered>::backward_buffered(
            linear, ctx, grad_output, grad_input, params, slice, grad_params_handle,
        );
    } else if let Some(relu) = layer.as_relu() {
        <ReLU as UniversalLayerBuffered>::backward_buffered(
            relu, ctx, grad_output, grad_input, params, slice, grad_params_handle,
        );
    } else if let Some(sigmoid) = layer.as_sigmoid() {
        <Sigmoid as UniversalLayerBuffered>::backward_buffered(
            sigmoid, ctx, grad_output, grad_input, params, slice, grad_params_handle,
        );
    } else if let Some(tanh) = layer.as_tanh() {
        <Tanh as UniversalLayerBuffered>::backward_buffered(
            tanh, ctx, grad_output, grad_input, params, slice, grad_params_handle,
        );
    } else if let Some(leaky) = layer.as_leaky_relu() {
        <LeakyReLU as UniversalLayerBuffered>::backward_buffered(
            leaky, ctx, grad_output, grad_input, params, slice, grad_params_handle,
        );
    } else if let Some(identity) = layer.as_identity() {
        <Identity as UniversalLayerBuffered>::backward_buffered(
            identity, ctx, grad_output, grad_input, params, slice, grad_params_handle,
        );
    } else if let Some(softmax) = layer.as_softmax() {
        <Softmax as UniversalLayerBuffered>::backward_buffered(
            softmax, ctx, grad_output, grad_input, params, slice, grad_params_handle,
        );
    } else if let Some(memory) = layer.as_memory() {
        <Memory as UniversalLayerBuffered>::backward_buffered(
            memory, ctx, grad_output, grad_input, params, slice, grad_params_handle,
        );
    } else if let Some(soft_sparse) = layer.as_soft_sparse_gate() {
        <SoftSparseGate as UniversalLayerBuffered>::backward_buffered(
            soft_sparse, ctx, grad_output, grad_input, params, slice, grad_params_handle,
        );
    } else if let Some(soft_keep) = layer.as_soft_keep_gate() {
        <SoftKeepGate as UniversalLayerBuffered>::backward_buffered(
            soft_keep, ctx, grad_output, grad_input, params, slice, grad_params_handle,
        );
    } else if let Some(dual_anchor) = layer.as_dual_anchor() {
        <DualAnchor as UniversalLayerBuffered>::backward_buffered(
            dual_anchor, ctx, grad_output, grad_input, params, slice, grad_params_handle,
        );
    } else {
        unreachable!(
            "Layer {:?} does not have a buffered backward implementation",
            std::any::type_name_of_val(layer.as_ref())
        );
    }
}