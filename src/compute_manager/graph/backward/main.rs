// src/compute_manager/graph/backward/main.rs

use std::time::Instant;
use faer::Mat;

use crate::compute_manager::dim_change;
use crate::compute_manager::matrix_buffer::{MatrixBuffer, TempMatrixPool};
use crate::compute_manager::graph::model::MixedModel;
use crate::compute_manager::graph::types::{DynamicContext, Segment};
use crate::compute_manager::persistent_buffer::SegmentPersistentBuffers;
use crate::compute_manager::gpu::processor::process_backward_gpu_buffered;
use crate::device_plan::plan::ComputeDevice;
use crate::layers::{
    UniversalLayer, UniversalLayerBuffered,
    Linear, ReLU, Sigmoid, Tanh, LeakyReLU, Identity, Softmax,
};
use crate::model_plan::param_store::ParamSlice;

impl MixedModel {
    /// Обратный матричный проход с множественными выходами (градиентами по выходам) и входами.
    /// Возвращает градиенты по входам модели и накопленные градиенты параметров.
    ///
    /// Для каждого сегмента фиксируется время выполнения и записывается
    /// в профилировочные данные адаптивного планировщика.
    pub fn backward_mat_multi(
        &mut self,
        contexts: &[Vec<DynamicContext>],
        deltas: &[Mat<f32>],
    ) -> (Vec<Mat<f32>>, Vec<Vec<f32>>) {
        assert_eq!(
            deltas.len(),
            self.output_stream_count,
            "backward_mat_multi: expected {} deltas, got {}",
            self.output_stream_count,
            deltas.len()
        );

        let params = self.store.lock().unwrap().all_params().to_vec();
        let param_len = params.len();
        let mut total_grad = vec![0.0f32; param_len];

        let mut stream_gradients: Vec<Mat<f32>> = deltas.to_vec();

        let total_context_len = contexts.first().map(|c| c.len()).unwrap_or(0);
        let mut ctx_pos = total_context_len;

        let segments = self.segments.clone();

        for (seg_index, seg) in segments.iter().enumerate().rev() {
            let start = Instant::now();

            match seg {
                Segment::Unsqueeze(target_dims) => {
                    for mat in &mut stream_gradients {
                        *mat = dim_change::reduce_mat(mat, target_dims);
                    }
                }
                Segment::ReduceMean(target_dims) => {
                    for mat in &mut stream_gradients {
                        *mat = dim_change::unsqueeze_mat(mat, target_dims);
                    }
                }
                Segment::UniversalProcessor(proc, slices, stream_indices) => {
                    let num_layers = proc.len();
                    let active_indices: Vec<usize> = match stream_indices {
                        Some(indices) => indices.clone(),
                        None => (0..stream_gradients.len()).collect(),
                    };

                    let mut new_gradients: Vec<Option<Mat<f32>>> = vec![None; stream_gradients.len()];

                    for &stream_idx in &active_indices {
                        let delta_mat = stream_gradients[stream_idx].clone();
                        let pos_in_sorted = active_indices.iter().position(|&x| x == stream_idx).unwrap();
                        let stream_ctx_start = ctx_pos - (active_indices.len() - pos_in_sorted) * num_layers;
                        let layer_ctxs: Vec<&DynamicContext> = contexts[0]
                            [stream_ctx_start..stream_ctx_start + num_layers]
                            .iter()
                            .collect();

                        let ctxs_owned: Vec<DynamicContext> = layer_ctxs.iter().map(|&c| c.clone()).collect();
                        let ctxs_slice: &[DynamicContext] = &ctxs_owned;

                        let in_delta_mat = if self.gpu_compute.is_some() {
                            let gpu = self.gpu_compute.as_ref().unwrap().lock().unwrap();
                            let segment_buffers = self.get_segment_buffers(seg_index);
                            if let Some(ref buffers) = segment_buffers {
                                crate::compute_manager::gpu::processor::process_backward_gpu(
                                    &gpu,
                                    buffers,
                                    proc,
                                    slices,
                                    ctxs_slice,
                                    &params,
                                    &delta_mat,
                                    &mut total_grad,
                                )
                            } else {
                                Self::backward_universal_batch_mat(
                                    proc,
                                    slices,
                                    &layer_ctxs,
                                    &delta_mat,
                                    &params,
                                    &mut total_grad,
                                )
                            }
                        } else {
                            Self::backward_universal_batch_mat(
                                proc,
                                slices,
                                &layer_ctxs,
                                &delta_mat,
                                &params,
                                &mut total_grad,
                            )
                        };

                        new_gradients[stream_idx] = Some(in_delta_mat);
                    }

                    for (i, opt) in new_gradients.iter_mut().enumerate() {
                        if opt.is_none() {
                            *opt = Some(stream_gradients[i].clone());
                        }
                    }
                    stream_gradients = new_gradients.into_iter().map(|o| o.unwrap()).collect();

                    ctx_pos -= num_layers * active_indices.len();
                }
                Segment::SplitterConnector { dim_a, dim_b } => {
                    assert!(stream_gradients.len() == 2);
                    let delta_a = stream_gradients[0].clone();
                    let delta_b = stream_gradients[1].clone();

                    let connector = crate::layers::SplitterConnector::new(*dim_a, *dim_b);
                    let dummy_ctx = DynamicContext::Mat(
                        crate::layers::mat_context::MatContext::SplitterConnector {
                            input: Mat::zeros(0, 0),
                        },
                    );
                    let (in_a, in_b, _) = connector.backward_mat(&dummy_ctx, &delta_a, &delta_b);
                    stream_gradients = vec![in_a, in_b];
                    ctx_pos -= 1;
                }
                Segment::CombinerConnector { input_dims: _, .. } => {
                    for mat in &mut stream_gradients {
                        let connector = crate::layers::CombinerConnector::new(vec![]);
                        let dummy_ctx = DynamicContext::Mat(
                            crate::layers::mat_context::MatContext::CombinerConnector {
                                inputs: vec![Mat::zeros(0, 0)],
                            },
                        );
                        let (in_mat, _) = connector.backward_mat(&dummy_ctx, mat);
                        *mat = in_mat;
                    }
                    ctx_pos -= 1;
                }
                Segment::Splitter {
                    input_dim,
                    output_dims,
                    slice,
                } => {
                    assert!(ctx_pos > 0);
                    let ctx = &contexts[0][ctx_pos - 1];
                    let (x_mat, pre_a_mat, pre_b_mat) = match ctx {
                        DynamicContext::Mat(
                            crate::layers::mat_context::MatContext::Splitter {
                                input,
                                pre_a,
                                pre_b,
                            },
                        ) => (input.clone(), pre_a.clone(), pre_b.clone()),
                        _ => panic!("Expected Splitter context"),
                    };

                    let da_mat = stream_gradients[0].clone();
                    let db_mat = stream_gradients[1].clone();

                    let (wa, wb, _, _) =
                        crate::layers::Splitter::new(*input_dim, output_dims.clone())
                            .get_weights_and_biases(&params, slice);

                    let (dx_mat, grad) = if let Some(ref gpu_compute_mutex) = self.gpu_compute {
                        let gpu = gpu_compute_mutex.lock().unwrap();
                        gpu.run_splitter_backward(
                            &x_mat,
                            &da_mat,
                            &db_mat,
                            &pre_a_mat,
                            &pre_b_mat,
                            &wa,
                            &wb,
                        )
                    } else {
                        crate::layers::Splitter::new(*input_dim, output_dims.clone()).backward_mat(
                            &x_mat,
                            &da_mat,
                            &db_mat,
                            &pre_a_mat,
                            &pre_b_mat,
                            &wa,
                            &wb,
                        )
                    };

                    for (idx, &g) in grad.iter().enumerate() {
                        total_grad[slice.start + idx] += g;
                    }

                    stream_gradients = vec![dx_mat];
                    ctx_pos -= 1;
                }
                Segment::Combiner {
                    input_dim,
                    output_dim,
                    slice,
                } => {
                    assert!(ctx_pos > 0);
                    let ctx = &contexts[0][ctx_pos - 1];
                    let (a_mat, b_mat, pre_mat) = match ctx {
                        DynamicContext::Mat(
                            crate::layers::mat_context::MatContext::Combiner {
                                input_a,
                                input_b,
                                pre_act,
                            },
                        ) => (input_a.clone(), input_b.clone(), pre_act.clone()),
                        _ => panic!("Expected Combiner context"),
                    };

                    let dout_mat = stream_gradients[0].clone();

                    let combiner =
                        crate::layers::Combiner::new(vec![*input_dim, *input_dim], *output_dim);
                    let (wa, wb, _) = combiner.get_weights_and_bias(&params, slice);
                    let (da_mat, db_mat, grad) = if let Some(ref gpu_compute_mutex) = self.gpu_compute
                    {
                        let gpu = gpu_compute_mutex.lock().unwrap();
                        gpu.run_combiner_backward(
                            &a_mat,
                            &b_mat,
                            &dout_mat,
                            &pre_mat,
                            &wa,
                            &wb,
                        )
                    } else {
                        combiner.backward_mat(&a_mat, &b_mat, &dout_mat, &params, slice)
                    };

                    for (idx, &g) in grad.iter().enumerate() {
                        total_grad[slice.start + idx] += g;
                    }

                    stream_gradients = vec![da_mat, db_mat];
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

        assert_eq!(
            stream_gradients.len(),
            self.input_stream_count,
            "backward_mat_multi: input stream count mismatch"
        );

        (stream_gradients, vec![total_grad])
    }

    pub fn backward_mat(
        &mut self,
        contexts: &[Vec<DynamicContext>],
        delta: &Mat<f32>,
    ) -> (Mat<f32>, Vec<Vec<f32>>) {
        let (ins, grads) = self.backward_mat_multi(contexts, &[delta.clone()]);
        assert_eq!(ins.len(), 1);
        (ins.into_iter().next().unwrap(), grads)
    }

    fn backward_universal_batch_mat(
        layers: &[Box<dyn UniversalLayer>],
        slices: &[ParamSlice],
        ctxs: &[&DynamicContext],
        delta: &Mat<f32>,
        params: &[f32],
        total_grad: &mut Vec<f32>,
    ) -> Mat<f32> {
        let mut current_delta = delta.clone();
        for i in (0..layers.len()).rev() {
            let (in_delta, grad) =
                layers[i].backward_mat(ctxs[i], &current_delta, params, &slices[i]);
            current_delta = in_delta;
            for (idx, &g) in grad.iter().enumerate() {
                total_grad[idx] += g;
            }
        }
        current_delta
    }

    // ===================================================================
    // Буферизованная версия обратного прохода (MatrixBuffer + TempMatrixPool)
    // ===================================================================

    /// Обратный проход с использованием пула временных матриц.
    /// Принимает контексты (пока `DynamicContext` с `Mat<f32>`),
    /// градиенты выходов как `Vec<MatrixBuffer>` и возвращает градиенты входов и накопленные градиенты параметров.
    pub fn backward_mat_multi_buffered(
        &mut self,
        pool: &mut TempMatrixPool,
        contexts: &[Vec<DynamicContext>],
        deltas: Vec<MatrixBuffer>,
    ) -> (Vec<MatrixBuffer>, Vec<Vec<f32>>) {
        assert_eq!(deltas.len(), self.output_stream_count,
            "backward_mat_multi_buffered: expected {} deltas, got {}",
            self.output_stream_count, deltas.len());

        let params = self.store.lock().unwrap().all_params().to_vec();
        let param_len = params.len();
        let mut total_grad = vec![0.0f32; param_len];

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
                        new_stream.push(dim_change::reduce_mat_buffered(pool, buf, target_dims));
                    }
                    stream_gradients = new_stream;
                }
                Segment::ReduceMean(target_dims) => {
                    let mut new_stream = Vec::with_capacity(stream_gradients.len());
                    for buf in stream_gradients {
                        new_stream.push(dim_change::unsqueeze_mat_buffered(pool, buf, target_dims));
                    }
                    stream_gradients = new_stream;
                }
                Segment::UniversalProcessor(proc, slices, stream_indices) => {
                    let num_layers = proc.len();
                    let active_indices: Vec<usize> = match stream_indices {
                        Some(indices) => indices.clone(),
                        None => (0..stream_gradients.len()).collect(),
                    };

                    let mut new_gradients: Vec<Option<MatrixBuffer>> =
                        (0..stream_gradients.len()).map(|_| None).collect();

                    for &stream_idx in &active_indices {
                        let delta_buf = std::mem::replace(
                            &mut stream_gradients[stream_idx],
                            MatrixBuffer::dummy(pool),
                        );
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

                            // Загружаем градиент на GPU
                            let delta_mat = delta_buf.to_mat();
                            let delta_gpu = gpu.upload_mat_to_gpu_matrix(&delta_mat);

                            // Получаем persistent buffers для сегмента
                            let segment_buffers_opt = self.get_segment_buffers(seg_index);
                            let temp_buffers;
                            let segment_buffers = if let Some(b) = segment_buffers_opt {
                                b
                            } else {
                                temp_buffers = SegmentPersistentBuffers::for_segment(
                                    seg,
                                    &self.segment_placement[seg_index].compute_device,
                                    delta_buf.rows(),
                                    &mut self.memory_executor.lock().unwrap(),
                                );
                                temp_buffers
                            };

                            let out_gpu = process_backward_gpu_buffered(
                                &gpu,
                                &segment_buffers,
                                proc,
                                slices,
                                ctxs_slice,
                                &params,
                                delta_gpu,
                                &mut total_grad,
                            );

                            // Конвертируем результат обратно в CPU MatrixBuffer
                            let out_mat = gpu.download_gpu_matrix_to_mat(&out_gpu);
                            let mut cpu_buf = pool.acquire(out_mat.nrows(), out_mat.ncols());
                            cpu_buf.copy_from_mat(&out_mat);
                            new_gradients[stream_idx] = Some(cpu_buf);
                        } else {
                            // CPU-путь
                            let in_delta_buf = self.backward_universal_batch_buffered(
                                pool,
                                proc,
                                slices,
                                &layer_ctxs,
                                delta_buf,
                                &params,
                                &mut total_grad,
                            );
                            new_gradients[stream_idx] = Some(in_delta_buf);
                        }
                    }

                    // Подставляем результаты в stream_gradients
                    let mut final_grads = Vec::with_capacity(stream_gradients.len());
                    for i in 0..stream_gradients.len() {
                        if let Some(buf) = new_gradients[i].take() {
                            final_grads.push(buf);
                        } else {
                            final_grads.push(std::mem::replace(
                                &mut stream_gradients[i],
                                MatrixBuffer::dummy(pool),
                            ));
                        }
                    }
                    stream_gradients = final_grads;

                    ctx_pos -= num_layers * active_indices.len();
                }
                Segment::SplitterConnector { dim_a, dim_b } => {
                    assert_eq!(stream_gradients.len(), 2);
                    let delta_a = std::mem::replace(&mut stream_gradients[0], MatrixBuffer::dummy(pool));
                    let delta_b = std::mem::replace(&mut stream_gradients[1], MatrixBuffer::dummy(pool));

                    // Коннектор просто пропускает градиенты без изменений.
                    let mut in_a = pool.acquire(delta_a.rows(), delta_a.cols());
                    let mut in_b = pool.acquire(delta_b.rows(), delta_b.cols());
                    in_a.copy_from_slice(delta_a.as_slice());
                    in_b.copy_from_slice(delta_b.as_slice());

                    pool.release(delta_a);
                    pool.release(delta_b);

                    stream_gradients = vec![in_a, in_b];
                    ctx_pos -= 1;
                }
                Segment::CombinerConnector { .. } => {
                    // Прозрачный проход: градиенты не меняются.
                    ctx_pos -= 1;
                }
                Segment::Splitter {
                    input_dim,
                    output_dims,
                    slice,
                } => {
                    assert!(ctx_pos > 0);
                    let ctx = &contexts[0][ctx_pos - 1];
                    let (x_mat, pre_a_mat, pre_b_mat) = match ctx {
                        DynamicContext::Mat(
                            crate::layers::mat_context::MatContext::Splitter {
                                input,
                                pre_a,
                                pre_b,
                            },
                        ) => (input.clone(), pre_a.clone(), pre_b.clone()),
                        _ => panic!("Expected Splitter context"),
                    };

                    let da_buf = std::mem::replace(&mut stream_gradients[0], MatrixBuffer::dummy(pool));
                    let db_buf = std::mem::replace(&mut stream_gradients[1], MatrixBuffer::dummy(pool));

                    let splitter = crate::layers::Splitter::new(*input_dim, output_dims.clone());
                    let (wa, wb, _, _) = splitter.get_weights_and_biases(&params, slice);

                    // Временно используем Mat для вычислений (до полного перехода на буферы в Этапе 2)
                    let da_mat = da_buf.to_mat();
                    let db_mat = db_buf.to_mat();

                    let (dx_mat, grad) = splitter.backward_mat(
                        &x_mat,
                        &da_mat,
                        &db_mat,
                        &pre_a_mat,
                        &pre_b_mat,
                        &wa,
                        &wb,
                    );

                    for (idx, &g) in grad.iter().enumerate() {
                        total_grad[slice.start + idx] += g;
                    }

                    let rows = dx_mat.nrows();
                    let cols = dx_mat.ncols();
                    let mut dx_buf = pool.acquire(rows, cols);
                    dx_buf.copy_from_mat(&dx_mat);

                    pool.release(da_buf);
                    pool.release(db_buf);

                    stream_gradients = vec![dx_buf];
                    ctx_pos -= 1;
                }
                Segment::Combiner {
                    input_dim,
                    output_dim,
                    slice,
                } => {
                    assert!(ctx_pos > 0);
                    let ctx = &contexts[0][ctx_pos - 1];
                    let (a_mat, b_mat, pre_mat) = match ctx {
                        DynamicContext::Mat(
                            crate::layers::mat_context::MatContext::Combiner {
                                input_a,
                                input_b,
                                pre_act,
                            },
                        ) => (input_a.clone(), input_b.clone(), pre_act.clone()),
                        _ => panic!("Expected Combiner context"),
                    };

                    let dout_buf = std::mem::replace(&mut stream_gradients[0], MatrixBuffer::dummy(pool));

                    let combiner = crate::layers::Combiner::new(vec![*input_dim, *input_dim], *output_dim);
                    let (wa, wb, _) = combiner.get_weights_and_bias(&params, slice);

                    let dout_mat = dout_buf.to_mat();
                    let (da_mat, db_mat, grad) = combiner.backward_mat(
                        &a_mat,
                        &b_mat,
                        &dout_mat,
                        &params,
                        slice,
                    );

                    for (idx, &g) in grad.iter().enumerate() {
                        total_grad[slice.start + idx] += g;
                    }

                    let rows_a = da_mat.nrows();
                    let cols_a = da_mat.ncols();
                    let mut da_buf = pool.acquire(rows_a, cols_a);
                    da_buf.copy_from_mat(&da_mat);

                    let rows_b = db_mat.nrows();
                    let cols_b = db_mat.ncols();
                    let mut db_buf = pool.acquire(rows_b, cols_b);
                    db_buf.copy_from_mat(&db_mat);

                    pool.release(dout_buf);

                    stream_gradients = vec![da_buf, db_buf];
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
        (stream_gradients, vec![total_grad])
    }

    /// Универсальный обратный проход через слои с использованием MatrixBuffer
    fn backward_universal_batch_buffered(
        &mut self,
        pool: &mut TempMatrixPool,
        layers: &[Box<dyn UniversalLayer>],
        slices: &[ParamSlice],
        ctxs: &[&DynamicContext],
        grad_out: MatrixBuffer,
        params: &[f32],
        total_grad: &mut Vec<f32>,
    ) -> MatrixBuffer {
        let mut current_grad = grad_out;
        for i in (0..layers.len()).rev() {
            let layer = &layers[i];
            let slice = &slices[i];
            let ctx = ctxs[i];

            // Явно разрешаем неоднозначность input_features через трейт UniversalLayer
            let in_features = if let Some(l) = layer.as_linear() {
                <dyn UniversalLayer>::input_features(l)
            } else if let Some(l) = layer.as_relu() {
                <dyn UniversalLayer>::input_features(l)
            } else if let Some(l) = layer.as_sigmoid() {
                <dyn UniversalLayer>::input_features(l)
            } else if let Some(l) = layer.as_tanh() {
                <dyn UniversalLayer>::input_features(l)
            } else if let Some(l) = layer.as_leaky_relu() {
                <dyn UniversalLayer>::input_features(l)
            } else if let Some(l) = layer.as_identity() {
                <dyn UniversalLayer>::input_features(l)
            } else if let Some(l) = layer.as_softmax() {
                <dyn UniversalLayer>::input_features(l)
            } else {
                // fallback – используем число столбцов текущего градиента
                current_grad.cols()
            };

            // Для слоёв без параметров (ReLU и т.п.) input_features возвращает 0,
            // но реальная размерность определяется current_grad. Поэтому подменяем.
            let real_in_features = if in_features == 0 { current_grad.cols() } else { in_features };
            let batch = current_grad.rows();
            let mut grad_input = pool.acquire(batch, real_in_features);

            let grad_params = if let Some(linear) = layer.as_linear() {
                <Linear as UniversalLayerBuffered>::backward_buffered(
                    linear, ctx, &current_grad, &mut grad_input, params, slice
                )
            } else if let Some(relu) = layer.as_relu() {
                <ReLU as UniversalLayerBuffered>::backward_buffered(
                    relu, ctx, &current_grad, &mut grad_input, params, slice
                )
            } else if let Some(sigmoid) = layer.as_sigmoid() {
                <Sigmoid as UniversalLayerBuffered>::backward_buffered(
                    sigmoid, ctx, &current_grad, &mut grad_input, params, slice
                )
            } else if let Some(tanh) = layer.as_tanh() {
                <Tanh as UniversalLayerBuffered>::backward_buffered(
                    tanh, ctx, &current_grad, &mut grad_input, params, slice
                )
            } else if let Some(leaky) = layer.as_leaky_relu() {
                <LeakyReLU as UniversalLayerBuffered>::backward_buffered(
                    leaky, ctx, &current_grad, &mut grad_input, params, slice
                )
            } else if let Some(identity) = layer.as_identity() {
                <Identity as UniversalLayerBuffered>::backward_buffered(
                    identity, ctx, &current_grad, &mut grad_input, params, slice
                )
            } else if let Some(softmax) = layer.as_softmax() {
                <Softmax as UniversalLayerBuffered>::backward_buffered(
                    softmax, ctx, &current_grad, &mut grad_input, params, slice
                )
            } else {
                // fallback на старый метод
                let (dx, grad) = layer.backward_mat(ctx, &current_grad.to_mat(), params, slice);
                grad_input.copy_from_mat(&dx);
                grad
            };

            for (idx, &g) in grad_params.iter().enumerate() {
                total_grad[slice.start + idx] += g;
            }

            pool.release(current_grad);
            current_grad = grad_input;
        }
        current_grad
    }
}