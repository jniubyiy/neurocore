// src/compute_manager/device_assignment.rs

use crate::compute_manager::graph::types::Segment;
use crate::device_plan::{ComputeDevice, DevicePlan, StorageDevice};
use crate::layers::UniversalLayer;

/// Информация о размещении одного сегмента модели.
#[derive(Debug, Clone)]
pub struct SegmentPlacement {
    /// Индекс сегмента в списке segments модели.
    pub segment_index: usize,
    /// На каком вычислительном устройстве исполняется сегмент.
    pub compute_device: ComputeDevice,
    /// Где хранятся параметры этого сегмента (если есть).
    /// None — если сегмент не имеет обучаемых параметров
    /// (например, ReLU, Sigmoid, Unsqueeze, коннекторы).
    pub parameter_storage: Option<StorageDevice>,
}

/// Автоматически распределяет устройства для всех сегментов модели,
/// основываясь на доступных устройствах и характеристиках слоёв.
///
/// Алгоритм (упрощённый):
/// - Если GPU присутствуют, то большие линейные слои (много параметров)
///   отправляются на GPU, их параметры — в VRAM;
///   маленькие линейные слои и все активации — на CPU, параметры в RAM.
/// - Если GPU нет, всё на первом CPU.
/// - Сегменты без параметров (ReLU, Unsqueeze и т.п.) выполняются на том же устройстве,
///   что и предыдущий значимый сегмент (для минимизации передач).
/// - Сплиттеры/комбайнеры: пока остаются на CPU (упрощение).
///
/// Возвращает вектор размещения, по одному элементу на сегмент.
pub fn assign_devices(
    segments: &[Segment],
    device_plan: &DevicePlan,
) -> Vec<SegmentPlacement> {
    let has_gpu = device_plan.compute_devices.iter().any(|d| matches!(d, ComputeDevice::Gpu { .. }));
    let default_cpu = device_plan.compute_devices.iter()
        .find_map(|d| if let ComputeDevice::Cpu { id, .. } = d { Some(ComputeDevice::Cpu { id: *id, threads: 0 }) } else { None })
        .unwrap_or(ComputeDevice::Cpu { id: 0, threads: 1 });
    let first_gpu = device_plan.compute_devices.iter()
        .find_map(|d| if let ComputeDevice::Gpu { id } = d { Some(ComputeDevice::Gpu { id: *id }) } else { None });

    // Хранилище по умолчанию для CPU параметров: первое доступное RAM
    let ram_storage = device_plan.storage_devices.iter()
        .find_map(|s| if let StorageDevice::Ram { id, max_mb } = s { Some(StorageDevice::Ram { id: *id, max_mb: *max_mb }) } else { None })
        .unwrap_or(StorageDevice::Ram { id: 0, max_mb: 8192 });

    // Хранилище для GPU параметров: первое VRAM, привязанное к тому же GPU
    fn find_vram_for_gpu(plan: &DevicePlan, gpu_id: usize) -> Option<StorageDevice> {
        plan.storage_devices.iter()
            .find_map(|s| if let StorageDevice::Vram { gpu_id: vgid, id, max_mb } = s {
                if *vgid == gpu_id { Some(StorageDevice::Vram { id: *id, gpu_id: *vgid, max_mb: *max_mb }) } else { None }
            } else { None })
    }

    let mut placements = Vec::with_capacity(segments.len());
    let mut last_compute = if let Some(ref gpu) = first_gpu { gpu.clone() } else { default_cpu.clone() };

    for (idx, segment) in segments.iter().enumerate() {
        match segment {
            Segment::UniversalProcessor(layers, _, _) => {
                let (use_gpu, has_params) = analyze_layers(layers, has_gpu, first_gpu.is_some());
                let compute = if use_gpu { first_gpu.clone().unwrap() } else { last_compute.clone() };
                let param_storage = if has_params {
                    if matches!(compute, ComputeDevice::Gpu { .. }) {
                        // Для GPU-слоя ищем VRAM, иначе запасной RAM
                        if let ComputeDevice::Gpu { id } = compute {
                            find_vram_for_gpu(device_plan, id)
                        } else { Some(ram_storage.clone()) }
                    } else {
                        Some(ram_storage.clone())
                    }
                } else {
                    None
                };
                placements.push(SegmentPlacement {
                    segment_index: idx,
                    compute_device: compute.clone(),
                    parameter_storage: param_storage,
                });
                last_compute = compute;
            }
            Segment::SplitterConnector { .. }
            | Segment::CombinerConnector { .. }
            | Segment::Splitter { .. }
            | Segment::Combiner { .. } => {
                // Пока всегда на CPU
                placements.push(SegmentPlacement {
                    segment_index: idx,
                    compute_device: default_cpu.clone(),
                    parameter_storage: None,
                });
                last_compute = default_cpu.clone();
            }
            Segment::Unsqueeze(_) | Segment::ReduceMean(_) => {
                // Операции изменения размерности – на том же устройстве, что и предыдущий слой
                placements.push(SegmentPlacement {
                    segment_index: idx,
                    compute_device: last_compute.clone(),
                    parameter_storage: None,
                });
            }
        }
    }

    placements
}

/// Анализирует список слоёв UniversalProcessor и возвращает (use_gpu, has_params).
/// use_gpu = true, если GPU доступен и хотя бы один слой содержит много параметров.
/// has_params = true, если есть обучаемые параметры.
fn analyze_layers(layers: &[Box<dyn UniversalLayer>], has_gpu: bool, gpu_available: bool) -> (bool, bool) {
    if !gpu_available || !has_gpu {
        return (false, layers.iter().any(|l| l.param_len() > 0));
    }
    let large_param_threshold = 1000; // байтов? или число параметров – эвристика
    let mut use_gpu = false;
    let mut has_params = false;
    for layer in layers.iter() {
        if layer.param_len() > 0 {
            has_params = true;
            if layer.param_len() > large_param_threshold {
                use_gpu = true;
            }
        }
    }
    // Если GPU есть, но все слои маленькие, то оставляем на CPU
    (use_gpu, has_params)
}