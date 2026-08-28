// src/compute_manager/compute_executor/strategy.rs

use crate::compute_manager::graph::types::Segment;
use crate::device_plan::plan::{ComputeDevice, DevicePlan, StorageDevice};

use super::placement::{SegmentPlacement, optimize_connectors};
use super::profiling::ProfilingState;

/// Вычисляет адаптивное размещение сегментов на основе накопленной профилировочной статистики.
///
/// Для каждого сегмента выбирается устройство с минимальной ожидаемой стоимостью выполнения.
/// Стоимость складывается из среднего времени (если есть измерения) и штрафа за текущую загрузку
/// устройства. При отсутствии данных используется эвристика на основе «тяжести» сегмента.
///
/// # Аргументы
/// * `segments` – список сегментов модели.
/// * `device_plan` – план устройств.
/// * `batch_size` – размер батча (не используется в текущей версии).
/// * `profiling` – накопленная профилировочная статистика.
/// * `current_placement` – текущее размещение (может использоваться для стабильности, пока игнорируется).
pub fn compute_adaptive_placement(
    segments: &[Segment],
    device_plan: &DevicePlan,
    _batch_size: usize,
    profiling: &ProfilingState,
    _current_placement: &[SegmentPlacement],
) -> Vec<SegmentPlacement> {
    let devices = &device_plan.compute_devices;
    if devices.is_empty() {
        return Vec::new();
    }

    let mut placements = Vec::with_capacity(segments.len());
    // Счётчики назначенных сегментов на каждое устройство (для оценки загрузки).
    let mut assigned_counts: std::collections::HashMap<ComputeDevice, usize> =
        devices.iter().map(|d| (d.clone(), 0)).collect();

    for (idx, segment) in segments.iter().enumerate() {
        let best_device = choose_best_device(segment, devices, profiling, &assigned_counts);
        // Обновляем счётчик загрузки выбранного устройства.
        *assigned_counts.get_mut(&best_device).unwrap() += 1;

        let parameter_storage = if segment_has_params(segment) {
            storage_for_compute_device(&best_device, device_plan)
        } else {
            None
        };

        placements.push(SegmentPlacement {
            segment_index: idx,
            compute_device: best_device,
            parameter_storage,
        });
    }

    // Выравниваем коннекторы и операции размерности к соседним вычислительным сегментам.
    optimize_connectors(segments, &mut placements);

    placements
}

/// Выбирает наилучшее устройство для сегмента с учётом профилирования и текущей загрузки.
fn choose_best_device(
    segment: &Segment,
    devices: &[ComputeDevice],
    profiling: &ProfilingState,
    assigned_counts: &std::collections::HashMap<ComputeDevice, usize>,
) -> ComputeDevice {
    let mut best_device = devices.first().cloned().unwrap();
    let mut best_cost = f64::MAX;

    for device in devices {
        let cost = estimate_cost(segment, device, profiling, assigned_counts);
        if cost < best_cost {
            best_cost = cost;
            best_device = device.clone();
        }
    }
    best_device
}

/// Оценивает стоимость выполнения сегмента на данном устройстве.
///
/// Возвращает значение в условных единицах: чем меньше, тем предпочтительнее.
/// Использует среднее время из профилировщика, если оно есть, иначе —
/// эвристическую оценку. К итоговой стоимости добавляется штраф за загрузку устройства.
fn estimate_cost(
    segment: &Segment,
    device: &ComputeDevice,
    profiling: &ProfilingState,
    assigned_counts: &std::collections::HashMap<ComputeDevice, usize>,
) -> f64 {
    // Базовое время (наносекунды)
    let base_time = if let Some(avg) = profiling.average_time(segment_index(segment, 0), device) {
        avg
    } else {
        fallback_time_estimate(segment, device)
    };

    // Штраф за загрузку: чем больше сегментов уже назначено на устройство,
    // тем больше стоимость. Коэффициент 0.2 (подбирается эмпирически).
    let load_count = assigned_counts.get(device).copied().unwrap_or(0) as f64;
    let load_penalty = load_count * base_time * 0.2;

    base_time + load_penalty
}

/// Эвристическая оценка времени выполнения для случая отсутствия профилировочных данных.
fn fallback_time_estimate(segment: &Segment, device: &ComputeDevice) -> f64 {
    // Определяем "тяжесть" сегмента в количестве операций (приблизительно).
    let heavy = match segment {
        Segment::UniversalProcessor(layers, _slices, _) => {
            let total_params: usize = layers.iter().map(|l| l.param_len()).sum();
            // Предполагаем, что каждый параметр добавляет ~10 наносекунд.
            total_params as f64 * 10.0 + 1000.0
        }
        Segment::Splitter { .. } | Segment::Combiner { .. } => 5000.0,
        _ => 200.0,
    };

    match device {
        ComputeDevice::Gpu { .. } => heavy * 0.3,   // GPU быстрее для тяжёлых сегментов
        ComputeDevice::Cpu { threads, .. } => heavy / (*threads as f64).max(1.0),
    }
}

/// Возвращает индекс сегмента. Вспомогательная функция, пока используется заглушка.
/// В реальной интеграции segment_index будет передаваться явно.
fn segment_index(_segment: &Segment, fallback: usize) -> usize {
    fallback
}

/// Проверяет, есть ли у сегмента обучаемые параметры.
fn segment_has_params(segment: &Segment) -> bool {
    match segment {
        Segment::UniversalProcessor(layers, _slices, _) => layers.iter().any(|l| l.param_len() > 0),
        Segment::Splitter { .. } | Segment::Combiner { .. } => true,
        _ => false,
    }
}

/// Определяет устройство хранения параметров для выбранного вычислительного устройства.
fn storage_for_compute_device(
    compute_device: &ComputeDevice,
    device_plan: &DevicePlan,
) -> Option<StorageDevice> {
    match compute_device {
        ComputeDevice::Gpu { id } => {
            device_plan.storage_devices.iter().find_map(|s| {
                if let StorageDevice::Vram { gpu_id, id: sid, max_mb } = s {
                    if *gpu_id == *id {
                        return Some(StorageDevice::Vram {
                            id: *sid,
                            gpu_id: *gpu_id,
                            max_mb: *max_mb,
                        });
                    }
                }
                None
            })
        }
        ComputeDevice::Cpu { .. } => {
            device_plan.storage_devices.iter().find_map(|s| {
                if let StorageDevice::Ram { id, max_mb } = s {
                    Some(StorageDevice::Ram {
                        id: *id,
                        max_mb: *max_mb,
                    })
                } else {
                    None
                }
            })
        }
    }
}