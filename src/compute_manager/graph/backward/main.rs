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
                        new_stream.push(dim_change::reduce_mat_buffered_handle(
                            &self.memory_executor,
                            pool,
                            buf,
                            target_dims,
                        ));
                    }
                    stream_gradients = new_stream;
                }
                Segment::ReduceMean(target_dims) => {
                    let mut new_stream = Vec::with_capacity(stream_gradients.len());
                    for buf in stream_gradients {
                        new_stream.push(dim_change::unsqueeze_mat_buffered_handle(
                            &self.memory_executor,
                            pool,
                            buf,
                            target_dims,
                        ));
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
                    {
                        let src_guard = delta_a.read();
                        let src = src_guard.as_slice().expect("CPU buffer");
                        let mut dst_guard = in_a.write();
                        let dst = dst_guard.as_slice_mut().expect("CPU buffer");
                        dst.copy_from_slice(src);
                    }
                    {
                        let src_guard = delta_b.read();
                        let src = src_guard.as_slice().expect("CPU buffer");
                        let mut dst_guard = in_b.write();
                        let dst = dst_guard.as_slice_mut().expect("CPU buffer");
                        dst.copy_from_slice(src);
                    }

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

                    // Читаем входные данные через MatrixBufferHandle::read()
                    let x_guard = x_handle.read();
                    let x_slice = x_guard.as_slice().expect("CPU buffer");
                    let da_guard = da_handle.read();
                    let da_slice = da_guard.as_slice().expect("CPU buffer");
                    let db_guard = db_handle.read();
                    let db_slice = db_guard.as_slice().expect("CPU buffer");
                    let pre_a_guard = pre_a_handle.read();
                    let pre_a_slice = pre_a_guard.as_slice().expect("CPU buffer");
                    let pre_b_guard = pre_b_handle.read();
                    let pre_b_slice = pre_b_guard.as_slice().expect("CPU buffer");

                    let splitter = crate::layers::Splitter::new(*input_dim, output_dims.clone());
                    let (wa_vec, wb_vec, _, _) = splitter.get_weights_and_biases_vec(&params, slice);

                    let batch = x_handle.rows();
                    let n = *input_dim;
                    let p = output_dims[0];
                    let q = output_dims[1];

                    // Создаём промежуточные управляемые буферы
                    let d_pre_a = pool.acquire(batch, p);
                    let d_pre_b = pool.acquire(batch, q);
                    let dx_handle = pool.acquire(batch, n);

                    // Заполняем d_pre_a и d_pre_b
                    {
                        let mut d_pre_a_guard = d_pre_a.write();
                        let d_pre_a_slice = d_pre_a_guard.as_slice_mut().expect("CPU buffer");
                        for i in 0..batch * p {
                            d_pre_a_slice[i] = if pre_a_slice[i] > 0.0 { da_slice[i] } else { 0.0 };
                        }
                    }
                    {
                        let mut d_pre_b_guard = d_pre_b.write();
                        let d_pre_b_slice = d_pre_b_guard.as_slice_mut().expect("CPU buffer");
                        for i in 0..batch * q {
                            d_pre_b_slice[i] = if pre_b_slice[i] > 0.0 { db_slice[i] } else { 0.0 };
                        }
                    }

                    // Получаем ссылки на d_pre_a и d_pre_b для вычислений
                    let d_pre_a_guard = d_pre_a.read();
                    let d_pre_a_slice = d_pre_a_guard.as_slice().expect("CPU buffer");
                    let d_pre_b_guard = d_pre_b.read();
                    let d_pre_b_slice = d_pre_b_guard.as_slice().expect("CPU buffer");

                    // Вычисляем dx и записываем в dx_handle
                    {
                        let mut dx_guard = dx_handle.write();
                        let dx_slice = dx_guard.as_slice_mut().expect("CPU buffer");
                        for r in 0..batch {
                            for c in 0..n {
                                let mut sum = 0.0;
                                for k in 0..p {
                                    sum += d_pre_a_slice[k * batch + r] * wa_vec[k * n + c];
                                }
                                for k in 0..q {
                                    sum += d_pre_b_slice[k * batch + r] * wb_vec[k * n + c];
                                }
                                dx_slice[c * batch + r] = sum;
                            }
                        }
                    }

                    // Записываем градиенты параметров в глобальный буфер
                    grad_params_handle.with_cpu_data_mut(|grad_data| {
                        // d_wa
                        for out_idx in 0..p {
                            for in_idx in 0..n {
                                let mut sum = 0.0;
                                for r in 0..batch {
                                    sum += d_pre_a_slice[out_idx * batch + r] * x_slice[in_idx * batch + r];
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
                                    sum += d_pre_b_slice[out_idx * batch + r] * x_slice[in_idx * batch + r];
                                }
                                grad_data[offset + out_idx * n + in_idx] = sum;
                            }
                        }

                        // d_bias_a
                        let offset = offset + q * n;
                        for c in 0..p {
                            let mut sum = 0.0;
                            for r in 0..batch {
                                sum += d_pre_a_slice[c * batch + r];
                            }
                            grad_data[offset + c] = sum;
                        }

                        // d_bias_b
                        let offset = offset + p;
                        for c in 0..q {
                            let mut sum = 0.0;
                            for r in 0..batch {
                                sum += d_pre_b_slice[c * batch + r];
                            }
                            grad_data[offset + c] = sum;
                        }
                    });

                    // Освобождаем временные буферы
                    pool.release(d_pre_a);
                    pool.release(d_pre_b);
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

                    // Читаем входные данные
                    let a_guard = a_handle.read();
                    let a_slice = a_guard.as_slice().expect("CPU buffer");
                    let b_guard = b_handle.read();
                    let b_slice = b_guard.as_slice().expect("CPU buffer");
                    let pre_guard = pre_handle.read();
                    let pre_slice = pre_guard.as_slice().expect("CPU buffer");
                    let dout_guard = dout_handle.read();
                    let dout_slice = dout_guard.as_slice().expect("CPU buffer");

                    let combiner = crate::layers::Combiner::new(vec![*input_dim, *input_dim], *output_dim);
                    let (wa_vec, wb_vec, _) = combiner.get_weights_and_bias_vec(&params, slice);

                    let batch = a_handle.rows();
                    let n = *input_dim;
                    let m = *output_dim;

                    // Промежуточные управляемые буферы
                    let d_pre_handle = pool.acquire(batch, m);
                    let da_handle = pool.acquire(batch, n);
                    let db_handle = pool.acquire(batch, n);

                    // Заполняем d_pre
                    {
                        let mut d_pre_guard = d_pre_handle.write();
                        let d_pre_slice = d_pre_guard.as_slice_mut().expect("CPU buffer");
                        for i in 0..batch * m {
                            d_pre_slice[i] = if pre_slice[i] > 0.0 { dout_slice[i] } else { 0.0 };
                        }
                    }

                    // Получаем ссылку на d_pre
                    let d_pre_guard = d_pre_handle.read();
                    let d_pre_slice = d_pre_guard.as_slice().expect("CPU buffer");

                    // Вычисляем da и db
                    {
                        let mut da_guard = da_handle.write();
                        let da_slice = da_guard.as_slice_mut().expect("CPU buffer");
                        let mut db_guard = db_handle.write();
                        let db_slice = db_guard.as_slice_mut().expect("CPU buffer");
                        for r in 0..batch {
                            for c in 0..n {
                                let mut sum_a = 0.0;
                                let mut sum_b = 0.0;
                                for k in 0..m {
                                    sum_a += d_pre_slice[k * batch + r] * wa_vec[k * n + c];
                                    sum_b += d_pre_slice[k * batch + r] * wb_vec[k * n + c];
                                }
                                da_slice[c * batch + r] = sum_a;
                                db_slice[c * batch + r] = sum_b;
                            }
                        }
                    }

                    // Записываем градиенты параметров
                    grad_params_handle.with_cpu_data_mut(|grad_data| {
                        // d_wa
                        for out_idx in 0..m {
                            for in_idx in 0..n {
                                let mut sum = 0.0;
                                for r in 0..batch {
                                    sum += d_pre_slice[out_idx * batch + r] * a_slice[in_idx * batch + r];
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
                                    sum += d_pre_slice[out_idx * batch + r] * b_slice[in_idx * batch + r];
                                }
                                grad_data[offset + out_idx * n + in_idx] = sum;
                            }
                        }

                        // d_bias
                        let offset = offset + m * n;
                        for c in 0..m {
                            let mut sum = 0.0;
                            for r in 0..batch {
                                sum += d_pre_slice[c * batch + r];
                            }
                            grad_data[offset + c] = sum;
                        }
                    });

                    // Освобождаем временные буферы
                    pool.release(d_pre_handle);
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