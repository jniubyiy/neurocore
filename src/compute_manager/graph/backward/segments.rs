// src/compute_manager/graph/backward/segments.rs

use std::time::Instant;
use faer::Mat;
use crate::compute_manager::dim_change;
use crate::compute_manager::graph::model::MixedModel;
use crate::device_plan::plan::ComputeDevice;

impl MixedModel {
    /// Обратный проход для операции Unsqueeze.
    /// Выполняет уменьшение размерности (reduce) над всеми потоковыми матрицами.
    /// При наличии GPU выполняет операцию на GPU, если сегмент размещён на GPU.
    pub(super) fn process_unsqueeze_backward(
        &mut self,
        stream_matrices: &mut Vec<Mat<f32>>,
        target_dims: &[usize],
        seg_index: usize,
    ) {
        let start = Instant::now();

        // Определяем устройство, на котором работает сегмент
        let device = self.segment_placement
            .get(seg_index)
            .map(|p| p.compute_device.clone())
            .unwrap_or(ComputeDevice::Cpu { id: 0, threads: 1 });

        match device {
            ComputeDevice::Gpu { id } => {
                // Используем GPU-реализацию, если доступен GpuCompute
                if let Some(ref gpu_compute_mutex) = self.gpu_compute {
                    let gpu = gpu_compute_mutex.lock().unwrap();
                    let segment_buffers = self.get_segment_buffers(seg_index);
                    for (i, mat) in stream_matrices.iter_mut().enumerate() {
                        // Входной persistent-буфер (выход предыдущего сегмента) содержит данные.
                        // Здесь предполагается, что данные для этого сегмента уже находятся в его входном буфере.
                        // Для простоты пока используем CPU-матрицу и загружаем её в GPU.
                        // В будущем можно передавать persistent-буферы напрямую.
                        // Сейчас выполняем reduce на GPU через временный буфер.
                        let reduced = crate::compute_manager::gpu::compute::dim_ops::reduce_mat_gpu(
                            &gpu,
                            mat,
                            target_dims,
                        );
                        *mat = reduced;
                    }
                } else {
                    // GPU не доступен, делаем CPU fallback
                    for mat in stream_matrices.iter_mut() {
                        *mat = dim_change::reduce_mat(mat, target_dims);
                    }
                }
            }
            _ => {
                // CPU-реализация
                for mat in stream_matrices.iter_mut() {
                    *mat = dim_change::reduce_mat(mat, target_dims);
                }
            }
        }

        let duration = start.elapsed().as_nanos() as f64;
        self.record_segment_timing(seg_index, &device, duration);
    }

    /// Обратный проход для операции ReduceMean.
    /// Выполняет увеличение размерности (unsqueeze) над всеми потоковыми матрицами.
    /// При наличии GPU выполняет операцию на GPU, если сегмент размещён на GPU.
    pub(super) fn process_reduce_mean_backward(
        &mut self,
        stream_matrices: &mut Vec<Mat<f32>>,
        target_dims: &[usize],
        seg_index: usize,
    ) {
        let start = Instant::now();

        let device = self.segment_placement
            .get(seg_index)
            .map(|p| p.compute_device.clone())
            .unwrap_or(ComputeDevice::Cpu { id: 0, threads: 1 });

        match device {
            ComputeDevice::Gpu { id } => {
                if let Some(ref gpu_compute_mutex) = self.gpu_compute {
                    let gpu = gpu_compute_mutex.lock().unwrap();
                    for mat in stream_matrices.iter_mut() {
                        let expanded = crate::compute_manager::gpu::compute::dim_ops::unsqueeze_mat_gpu(
                            &gpu,
                            mat,
                            target_dims,
                        );
                        *mat = expanded;
                    }
                } else {
                    for mat in stream_matrices.iter_mut() {
                        *mat = dim_change::unsqueeze_mat(mat, target_dims);
                    }
                }
            }
            _ => {
                for mat in stream_matrices.iter_mut() {
                    *mat = dim_change::unsqueeze_mat(mat, target_dims);
                }
            }
        }

        let duration = start.elapsed().as_nanos() as f64;
        self.record_segment_timing(seg_index, &device, duration);
    }
}