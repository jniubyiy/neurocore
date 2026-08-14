// src/compute_manager/graph/forward/main.rs

use std::time::Instant;
use faer::Mat;

use crate::compute_manager::dim_change;
use crate::compute_manager::matrix_buffer::{MatrixBufferHandle, TempMatrixPool};
use crate::compute_manager::graph::model::MixedModel;
use crate::compute_manager::graph::types::{DynamicContext, Segment};
use crate::compute_manager::persistent_buffer::SegmentPersistentBuffers;
use crate::compute_manager::gpu::processor::process_forward_gpu_buffered;
use crate::device_plan::plan::ComputeDevice;

impl MixedModel {
    /// Прямой матричный проход с множественными входами и выходами.
    /// Оставлен для обратной совместимости (GPU и старый CPU‑путь).
    pub fn forward_mat_multi(
        &mut self,
        inputs: &[Mat<f32>],
    ) -> (Vec<Mat<f32>>, Vec<Vec<DynamicContext>>) {
        assert_eq!(
            inputs.len(),
            self.input_stream_count,
            "forward_mat_multi: expected {} inputs, got {}",
            self.input_stream_count,
            inputs.len()
        );

        let batch_size = if let Some(first) = inputs.first() {
            first.nrows()
        } else {
            return (Vec::new(), Vec::new());
        };
        for mat in inputs {
            assert_eq!(
                mat.nrows(),
                batch_size,
                "All input matrices must have the same number of rows (batch size)"
            );
        }

        let mut stream_matrices: Vec<Mat<f32>> = inputs.to_vec();
        let mut all_ctxs: Vec<Vec<DynamicContext>> = vec![Vec::new(); batch_size];

        // Клонируем сегменты перед итерацией, чтобы избежать удержания
        // неизменяемой ссылки на self.segments при мутабельных вызовах self.process_*.
        let segments = self.segments.clone();

        for (seg_index, seg) in segments.iter().enumerate() {
            let start = Instant::now();

            match seg {
                Segment::Unsqueeze(target_dims) => {
                    for mat in &mut stream_matrices {
                        *mat = dim_change::unsqueeze_mat(mat, target_dims);
                    }
                }
                Segment::ReduceMean(target_dims) => {
                    for mat in &mut stream_matrices {
                        *mat = dim_change::reduce_mat(mat, target_dims);
                    }
                }
                Segment::UniversalProcessor(proc, slices, stream_indices) => {
                    let params = self.store.lock().unwrap().all_params();
                    self.process_universal_processor_forward(
                        proc,
                        slices,
                        seg_index,
                        &params,
                        &mut stream_matrices,
                        &mut all_ctxs,
                        stream_indices,
                    );
                }
                Segment::SplitterConnector { dim_a, dim_b } => {
                    self.process_splitter_connector_forward(
                        *dim_a,
                        *dim_b,
                        batch_size,
                        &mut stream_matrices,
                        &mut all_ctxs,
                        seg_index,
                    );
                }
                Segment::CombinerConnector { input_dims, .. } => {
                    self.process_combiner_connector_forward(
                        input_dims.clone(),
                        batch_size,
                        &mut stream_matrices,
                        &mut all_ctxs,
                        seg_index,
                    );
                }
                Segment::Splitter {
                    input_dim,
                    output_dims,
                    slice,
                } => {
                    self.process_splitter_forward(
                        *input_dim,
                        output_dims.clone(),
                        *slice,
                        batch_size,
                        &mut stream_matrices,
                        &mut all_ctxs,
                        seg_index,
                    );
                }
                Segment::Combiner {
                    input_dim,
                    output_dim,
                    slice,
                } => {
                    self.process_combiner_forward(
                        *input_dim,
                        *output_dim,
                        *slice,
                        batch_size,
                        &mut stream_matrices,
                        &mut all_ctxs,
                        seg_index,
                    );
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
            stream_matrices.len(),
            self.output_stream_count,
            "forward_mat_multi: output stream count mismatch"
        );

        (stream_matrices, all_ctxs)
    }

    pub fn forward_mat(
        &mut self,
        input: &Mat<f32>,
    ) -> (Mat<f32>, Vec<Vec<DynamicContext>>) {
        let (outs, ctxs) = self.forward_mat_multi(&[input.clone()]);
        assert_eq!(outs.len(), 1);
        (outs.into_iter().next().unwrap(), ctxs)
    }

    // -----------------------------------------------------------------------
    // Новый метод: прямой проход с управляемыми буферами через TempMatrixPool
    // -----------------------------------------------------------------------

    /// Прямой проход с использованием [`MatrixBufferHandle`] и пула [`TempMatrixPool`].
    ///
    /// Этот метод полностью заменяет `forward_mat_multi_buffered` в новом CPU/GPU‑пути.
    /// Все промежуточные матрицы выделяются через пул дескрипторов, что позволяет
    /// `MemoryExecutor` отслеживать и переиспользовать память.
    pub fn forward_mat_multi_buffered(
        &mut self,
        pool: &mut TempMatrixPool,
        inputs: Vec<MatrixBufferHandle>,
    ) -> (Vec<MatrixBufferHandle>, Vec<Vec<DynamicContext>>) {
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
        let mut all_ctxs: Vec<Vec<DynamicContext>> = vec![Vec::new(); batch_size];

        let segments = self.segments.clone();

        for (seg_index, seg) in segments.iter().enumerate() {
            let start = Instant::now();

            match seg {
                Segment::Unsqueeze(target_dims) => {
                    let mut new_stream = Vec::with_capacity(stream_buffers.len());
                    for buf in stream_buffers {
                        new_stream.push(dim_change::unsqueeze_mat_buffered_handle(pool, buf, target_dims));
                    }
                    stream_buffers = new_stream;
                }
                Segment::ReduceMean(target_dims) => {
                    let mut new_stream = Vec::with_capacity(stream_buffers.len());
                    for buf in stream_buffers {
                        new_stream.push(dim_change::reduce_mat_buffered_handle(pool, buf, target_dims));
                    }
                    stream_buffers = new_stream;
                }
                Segment::UniversalProcessor(proc, slices, stream_indices) => {
                    let active_indices: Vec<usize> = match stream_indices {
                        Some(indices) => indices.clone(),
                        None => (0..stream_buffers.len()).collect(),
                    };
                    let params = self.store.lock().unwrap().all_params();

                    if let Some(ref gpu_compute_mutex) = self.gpu_compute {
                        let gpu = gpu_compute_mutex.lock().unwrap();

                        // Получаем или создаём persistent buffers для сегмента
                        let segment_buffers_opt = self.get_segment_buffers(seg_index);
                        let temp_buffers;
                        let segment_buffers = if let Some(b) = segment_buffers_opt {
                            b
                        } else {
                            temp_buffers = SegmentPersistentBuffers::for_segment(
                                seg,
                                &self.segment_placement[seg_index].compute_device,
                                batch_size,
                                &mut self.memory_executor.lock().unwrap(),
                            );
                            temp_buffers
                        };

                        for &stream_idx in &active_indices {
                            let input_buf = stream_buffers[stream_idx].clone();

                            // Если входной буфер CPU, загружаем его на GPU
                            let input_gpu = if input_buf.is_gpu() {
                                input_buf
                            } else {
                                let mut gpu_buf = gpu.allocate_gpu_matrix_handle(
                                    input_buf.rows(),
                                    input_buf.cols(),
                                );
                                gpu.copy_cpu_to_gpu_handle(&input_buf, &gpu_buf);
                                gpu_buf
                            };

                            let (out_gpu, layer_ctxs) = process_forward_gpu_buffered(
                                &gpu,
                                &segment_buffers,
                                proc,
                                slices,
                                &params,
                                input_gpu,
                            );

                            // Обновляем stream_buffers
                            stream_buffers[stream_idx] = out_gpu;

                            // Добавляем контексты для всех сэмплов
                            for sample_ctxs in all_ctxs.iter_mut() {
                                sample_ctxs.extend(layer_ctxs.clone());
                            }
                        }
                    } else {
                        // CPU-путь
                        self.process_universal_processor_forward_buffered(
                            pool,
                            proc,
                            slices,
                            seg_index,
                            &params,
                            &mut stream_buffers,
                            &mut all_ctxs,
                            stream_indices,
                        );
                    }
                }
                Segment::SplitterConnector { dim_a, dim_b } => {
                    self.process_splitter_connector_forward_buffered(
                        pool,
                        *dim_a,
                        *dim_b,
                        batch_size,
                        &mut stream_buffers,
                        &mut all_ctxs,
                        seg_index,
                    );
                }
                Segment::CombinerConnector { input_dims, .. } => {
                    self.process_combiner_connector_forward_buffered(
                        pool,
                        input_dims.clone(),
                        batch_size,
                        &mut stream_buffers,
                        &mut all_ctxs,
                        seg_index,
                    );
                }
                Segment::Splitter {
                    input_dim,
                    output_dims,
                    slice,
                } => {
                    self.process_splitter_forward_buffered(
                        pool,
                        *input_dim,
                        output_dims.clone(),
                        *slice,
                        batch_size,
                        &mut stream_buffers,
                        &mut all_ctxs,
                        seg_index,
                    );
                }
                Segment::Combiner {
                    input_dim,
                    output_dim,
                    slice,
                } => {
                    self.process_combiner_forward_buffered(
                        pool,
                        *input_dim,
                        *output_dim,
                        *slice,
                        batch_size,
                        &mut stream_buffers,
                        &mut all_ctxs,
                        seg_index,
                    );
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
            stream_buffers.len(),
            self.output_stream_count,
            "forward_mat_multi_buffered: output stream count mismatch"
        );

        (stream_buffers, all_ctxs)
    }
}