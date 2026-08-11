// src/compute_manager/graph/backward/main.rs

use std::time::Instant;
use faer::Mat;
use crate::compute_manager::dim_change;
use crate::compute_manager::graph::model::MixedModel;
use crate::compute_manager::graph::types::{DynamicContext, Segment};
use crate::device_plan::plan::ComputeDevice;
use crate::layers::UniversalLayer;
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

        // Потоки градиентов: каждый элемент — матрица градиентов для одного потока
        let mut stream_gradients: Vec<Mat<f32>> = deltas.to_vec();

        let total_context_len = contexts.first().map(|c| c.len()).unwrap_or(0);
        let mut ctx_pos = total_context_len;

        // Клонируем сегменты, чтобы не удерживать неизменяемую ссылку на self.segments
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
                                    buffers,               // &SegmentPersistentBuffers
                                    proc,                  // &[Box<dyn UniversalLayer>]
                                    slices,                // &[ParamSlice]
                                    ctxs_slice,            // &[DynamicContext]
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
                Segment::CombinerConnector { input_dims, .. } => {
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

            // Запись времени выполнения сегмента
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

    /// Обратный матричный проход (один выход – один вход).
    /// Оставлен для обратной совместимости.
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
}