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
    pub fn backward_mat_multi_buffered(
        &mut self,
        pool: &mut TempMatrixPool,
        contexts: &[Vec<DynamicContext>],
        deltas: Vec<MatrixBufferHandle>,
    ) -> Vec<MatrixBufferHandle> {
        assert_eq!(deltas.len(), self.output_stream_count,
            "backward_mat_multi_buffered: expected {} deltas, got {}",
            self.output_stream_count, deltas.len());

        let bp = self.buffered_param_store.lock().unwrap();
        let params_handle = bp.params_handle().clone();
        let grad_params_handle = bp.grads_handle().clone();
        drop(bp);

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
                                &params_handle,
                                delta_gpu_handle,
                                &grad_params_handle,
                            );

                            // Новый способ: прямое копирование GPU->CPU без faer::Mat
                            let cpu_handle = pool.acquire(out_gpu.rows(), out_gpu.cols());
                            gpu.copy_gpu_to_cpu_handle(&out_gpu, &cpu_handle);

                            new_gradients[stream_idx] = Some(cpu_handle);
                        } else {
                            let in_delta_handle = self.backward_universal_batch_buffered_handle(
                                pool,
                                proc,
                                slices,
                                &layer_ctxs,
                                delta_handle,
                                &params_handle,
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
                        let mut mem = self.memory_executor.lock().unwrap();
                        mem.copy_cpu_buffer(delta_a.id(), in_a.id());
                        mem.copy_cpu_buffer(delta_b.id(), in_b.id());
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

                    let batch = x_handle.rows();
                    let n = *input_dim;
                    let p = output_dims[0];
                    let q = output_dims[1];

                    let dx_handle = pool.acquire(batch, n);

                    let ids = [
                        x_handle.id(), da_handle.id(), db_handle.id(),
                        pre_a_handle.id(), pre_b_handle.id(),
                        params_handle.id(), grad_params_handle.id(), dx_handle.id(),
                    ];

                    x_handle.memory().lock().unwrap().with_cpu_slices_mut(&ids, |slices| {
                        let (first, rest) = slices.split_at_mut(1);
                        let x: &[f32] = &*first[0];
                        let (second, rest) = rest.split_at_mut(1);
                        let da: &[f32] = &*second[0];
                        let (third, rest) = rest.split_at_mut(1);
                        let db: &[f32] = &*third[0];
                        let (fourth, rest) = rest.split_at_mut(1);
                        let pre_a: &[f32] = &*fourth[0];
                        let (fifth, rest) = rest.split_at_mut(1);
                        let pre_b: &[f32] = &*fifth[0];
                        let (sixth, rest) = rest.split_at_mut(1);
                        let params_ref: &[f32] = &*sixth[0];
                        let (seventh, eighth) = rest.split_at_mut(1);
                        let gp: &mut [f32] = &mut *seventh[0];
                        let dx_out: &mut [f32] = &mut *eighth[0];

                        let wa_start = slice.start;
                        let wa_len = p * n;
                        let wb_start = wa_start + wa_len;
                        let wb_len = q * n;
                        let bias_a_start = wb_start + wb_len;
                        let bias_b_start = bias_a_start + p;

                        // Вычисляем dx
                        for r in 0..batch {
                            for c in 0..n {
                                let mut sum = 0.0;
                                for k in 0..p {
                                    let d_pre_a_val = if pre_a[k * batch + r] > 0.0 { da[k * batch + r] } else { 0.0 };
                                    sum += d_pre_a_val * params_ref[wa_start + k * n + c];
                                }
                                for k in 0..q {
                                    let d_pre_b_val = if pre_b[k * batch + r] > 0.0 { db[k * batch + r] } else { 0.0 };
                                    sum += d_pre_b_val * params_ref[wb_start + k * n + c];
                                }
                                dx_out[c * batch + r] = sum;
                            }
                        }

                        // Градиенты весов
                        for out_idx in 0..p {
                            for in_idx in 0..n {
                                let mut sum = 0.0;
                                for r in 0..batch {
                                    let d_pre_a_val = if pre_a[out_idx * batch + r] > 0.0 { da[out_idx * batch + r] } else { 0.0 };
                                    sum += d_pre_a_val * x[in_idx * batch + r];
                                }
                                gp[wa_start + out_idx * n + in_idx] = sum;
                            }
                        }
                        for out_idx in 0..q {
                            for in_idx in 0..n {
                                let mut sum = 0.0;
                                for r in 0..batch {
                                    let d_pre_b_val = if pre_b[out_idx * batch + r] > 0.0 { db[out_idx * batch + r] } else { 0.0 };
                                    sum += d_pre_b_val * x[in_idx * batch + r];
                                }
                                gp[wb_start + out_idx * n + in_idx] = sum;
                            }
                        }

                        // Градиенты смещений
                        for c in 0..p {
                            let mut sum = 0.0;
                            for r in 0..batch {
                                let d_pre_a_val = if pre_a[c * batch + r] > 0.0 { da[c * batch + r] } else { 0.0 };
                                sum += d_pre_a_val;
                            }
                            gp[bias_a_start + c] = sum;
                        }
                        for c in 0..q {
                            let mut sum = 0.0;
                            for r in 0..batch {
                                let d_pre_b_val = if pre_b[c * batch + r] > 0.0 { db[c * batch + r] } else { 0.0 };
                                sum += d_pre_b_val;
                            }
                            gp[bias_b_start + c] = sum;
                        }
                    });

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

                    let batch = a_handle.rows();
                    let n = *input_dim;
                    let m = *output_dim;

                    let da_handle = pool.acquire(batch, n);
                    let db_handle = pool.acquire(batch, n);

                    let ids = [
                        a_handle.id(), b_handle.id(), pre_handle.id(), dout_handle.id(),
                        params_handle.id(), grad_params_handle.id(), da_handle.id(), db_handle.id(),
                    ];

                    a_handle.memory().lock().unwrap().with_cpu_slices_mut(&ids, |slices| {
                        let (first, rest) = slices.split_at_mut(1);
                        let a: &[f32] = &*first[0];
                        let (second, rest) = rest.split_at_mut(1);
                        let b: &[f32] = &*second[0];
                        let (third, rest) = rest.split_at_mut(1);
                        let pre: &[f32] = &*third[0];
                        let (fourth, rest) = rest.split_at_mut(1);
                        let dout: &[f32] = &*fourth[0];
                        let (fifth, rest) = rest.split_at_mut(1);
                        let params_ref: &[f32] = &*fifth[0];
                        let (sixth, rest) = rest.split_at_mut(1);
                        let gp: &mut [f32] = &mut *sixth[0];
                        let (seventh, eighth) = rest.split_at_mut(1);
                        let da_out: &mut [f32] = &mut *seventh[0];
                        let db_out: &mut [f32] = &mut *eighth[0];

                        let wa_start = slice.start;
                        let wa_len = m * n;
                        let wb_start = wa_start + wa_len;
                        let wb_len = m * n;
                        let bias_start = wb_start + wb_len;

                        // Вычисляем da и db
                        for r in 0..batch {
                            for c in 0..n {
                                let mut sum_a = 0.0;
                                let mut sum_b = 0.0;
                                for k in 0..m {
                                    let d_pre = if pre[k * batch + r] > 0.0 { dout[k * batch + r] } else { 0.0 };
                                    sum_a += d_pre * params_ref[wa_start + k * n + c];
                                    sum_b += d_pre * params_ref[wb_start + k * n + c];
                                }
                                da_out[c * batch + r] = sum_a;
                                db_out[c * batch + r] = sum_b;
                            }
                        }

                        // Градиенты весов
                        for out_idx in 0..m {
                            for in_idx in 0..n {
                                let mut sum = 0.0;
                                for r in 0..batch {
                                    let d_pre = if pre[out_idx * batch + r] > 0.0 { dout[out_idx * batch + r] } else { 0.0 };
                                    sum += d_pre * a[in_idx * batch + r];
                                }
                                gp[wa_start + out_idx * n + in_idx] = sum;
                            }
                        }
                        for out_idx in 0..m {
                            for in_idx in 0..n {
                                let mut sum = 0.0;
                                for r in 0..batch {
                                    let d_pre = if pre[out_idx * batch + r] > 0.0 { dout[out_idx * batch + r] } else { 0.0 };
                                    sum += d_pre * b[in_idx * batch + r];
                                }
                                gp[wb_start + out_idx * n + in_idx] = sum;
                            }
                        }

                        // Градиент смещения
                        for c in 0..m {
                            let mut sum = 0.0;
                            for r in 0..batch {
                                let d_pre = if pre[c * batch + r] > 0.0 { dout[c * batch + r] } else { 0.0 };
                                sum += d_pre;
                            }
                            gp[bias_start + c] = sum;
                        }
                    });

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

    fn backward_universal_batch_buffered_handle(
        &mut self,
        pool: &mut TempMatrixPool,
        layers: &[Box<dyn UniversalLayer>],
        slices: &[ParamSlice],
        ctxs: &[&DynamicContext],
        grad_out: MatrixBufferHandle,
        params: &MatrixBufferHandle,
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

fn call_backward_buffered(
    layer: &Box<dyn UniversalLayer>,
    ctx: &DynamicContext,
    grad_output: &MatrixBufferHandle,
    grad_input: &mut MatrixBufferHandle,
    params: &MatrixBufferHandle,
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