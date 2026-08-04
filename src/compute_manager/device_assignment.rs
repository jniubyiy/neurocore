// src/compute_manager/device_assignment.rs

use std::collections::HashMap;
use crate::compute_manager::device_spec::DeviceId;
use crate::compute_manager::graph::types::Segment;
use crate::compute_manager::memory_executor::{BufferPriority, MemoryDeviceKind, MemoryError, MemoryExecutor};
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

/// Оценка потребления памяти для сегмента (в количестве элементов f32).
#[derive(Debug, Clone, Copy)]
struct SegmentMemory {
    /// Память под параметры (веса, bias)
    pub param_elements: usize,
    /// Оценочная память под активации (промежуточные тензоры)
    pub activation_elements: usize,
    /// Общая память (с запасом)
    pub total_elements: usize,
}

impl SegmentMemory {
    /// Создаёт оценку с коэффициентом запаса 1.2 (20% сверх)
    pub fn with_safety_margin(param: usize, activation: usize) -> Self {
        let total = ((param + activation) as f32 * 1.2) as usize;
        SegmentMemory {
            param_elements: param,
            activation_elements: activation,
            total_elements: total,
        }
    }
}

/// Вычисляет потребление памяти для сегмента на основе его содержимого и размера батча.
fn calculate_segment_memory(seg: &Segment, batch_size: usize) -> SegmentMemory {
    match seg {
        Segment::UniversalProcessor(layers, slices, _) => {
            // Суммарные параметры из слайсов
            let param_sum: usize = slices.iter().map(|s| s.len).sum();
            // Оценка активаций: максимальный размер тензора в слоях
            let max_feat = layers
                .iter()
                .map(|l| l.input_features().max(l.output_features()))
                .max()
                .unwrap_or(0);
            let activation_est = max_feat * batch_size;
            SegmentMemory::with_safety_margin(param_sum, activation_est)
        }
        Segment::Splitter { input_dim, output_dims, .. } => {
            let p = output_dims[0];
            let q = output_dims[1];
            let param_sum = input_dim * p + input_dim * q + p + q;
            let activation_est = (input_dim + p + q) * batch_size;
            SegmentMemory::with_safety_margin(param_sum, activation_est)
        }
        Segment::Combiner { input_dim, output_dim, .. } => {
            let param_sum = 2 * input_dim * output_dim + output_dim;
            let activation_est = (input_dim * 2 + output_dim) * batch_size;
            SegmentMemory::with_safety_margin(param_sum, activation_est)
        }
        Segment::SplitterConnector { dim_a, dim_b } => {
            // Нет параметров, только активации (вход и два выхода)
            let activation_est = (dim_a + dim_b) * batch_size;
            SegmentMemory::with_safety_margin(0, activation_est)
        }
        Segment::CombinerConnector { input_dims, output_dim, .. } => {
            let activation_est = (input_dims.iter().sum::<usize>() + output_dim) * batch_size;
            SegmentMemory::with_safety_margin(0, activation_est)
        }
        Segment::Unsqueeze(_) | Segment::ReduceMean(_) => {
            // Операции изменения размерности – параметров нет, активации примерно равны входу
            // Оценим грубо: size = batch_size * среднее число элементов
            SegmentMemory::with_safety_margin(0, 0)
        }
    }
}

/// Находит наиболее подходящее вычислительное устройство для сегмента,
/// пытаясь зарезервировать на нём память. Возвращает `Some(ComputeDevice)` и тип хранилища,
/// если удалось зарезервировать.
fn try_assign_device(
    seg: &Segment,
    memory_need: SegmentMemory,
    device_plan: &DevicePlan,
    memory_executor: &mut MemoryExecutor,
    _last_compute: &ComputeDevice,
) -> Result<(ComputeDevice, Option<StorageDevice>), MemoryError> {
    let has_gpu = device_plan.compute_devices.iter().any(|d| matches!(d, ComputeDevice::Gpu { .. }));
    let default_cpu = device_plan
        .compute_devices
        .iter()
        .find_map(|d| {
            if let ComputeDevice::Cpu { id, threads } = d {
                Some(ComputeDevice::Cpu { id: *id, threads: *threads })
            } else {
                None
            }
        })
        .unwrap_or(ComputeDevice::Cpu { id: 0, threads: 1 });

    // Определяем, нужно ли сегменту GPU (если есть параметры и они большие)
    let (prefer_gpu, has_params) = match seg {
        Segment::UniversalProcessor(layers, _, _) => {
            let (use_gpu, hp) = analyze_layers(layers, has_gpu, device_plan.compute_devices.iter().any(|d| matches!(d, ComputeDevice::Gpu { .. })));
            (use_gpu, hp)
        }
        Segment::Splitter { .. } | Segment::Combiner { .. } => {
            // Обучаемые сплиттеры/комбайнеры – обычно на GPU, если есть
            (has_gpu, true)
        }
        _ => (false, false),
    };

    // Собираем кандидатов в порядке предпочтения
    let mut candidates = Vec::new();
    if prefer_gpu {
        // Сначала GPU
        for dev in &device_plan.compute_devices {
            if let ComputeDevice::Gpu { id } = dev {
                candidates.push((ComputeDevice::Gpu { id: *id }, MemoryDeviceKind::DeviceVram(DeviceId(*id))));
            }
        }
    }
    // Затем CPU
    for dev in &device_plan.compute_devices {
        if let ComputeDevice::Cpu { id, threads } = dev {
            candidates.push((ComputeDevice::Cpu { id: *id, threads: *threads }, MemoryDeviceKind::HostRam));
        }
    }
    // Если нет предпочтения GPU, но есть GPU и он не был добавлен, добавим его как fallback
    if !prefer_gpu && has_gpu {
        for dev in &device_plan.compute_devices {
            if let ComputeDevice::Gpu { id } = dev {
                if !candidates.iter().any(|(c, _)| matches!(c, ComputeDevice::Gpu { id: cid } if *cid == *id)) {
                    candidates.push((ComputeDevice::Gpu { id: *id }, MemoryDeviceKind::DeviceVram(DeviceId(*id))));
                }
            }
        }
    }
    // Если всё равно пусто – добавим default_cpu
    if candidates.is_empty() {
        candidates.push((default_cpu.clone(), MemoryDeviceKind::HostRam));
    }

    // Пытаемся зарезервировать память на каждом кандидате
    for (compute_device, mem_kind) in candidates {
        // Пропускаем, если устройство не подходит по архитектуре (например, нет GPU)
        if let ComputeDevice::Gpu { id } = &compute_device {
            if !device_plan.compute_devices.iter().any(|d| matches!(d, ComputeDevice::Gpu { id: did } if did == id)) {
                continue;
            }
        }
        // Проверяем, есть ли хранилище для параметров (если есть параметры)
        let param_storage = if has_params {
            match &compute_device {
                ComputeDevice::Gpu { id } => {
                    // Ищем VRAM, привязанную к этому GPU
                    device_plan.storage_devices.iter().find_map(|s| {
                        if let StorageDevice::Vram { gpu_id, id: sid, max_mb } = s {
                            if *gpu_id == *id {
                                Some(StorageDevice::Vram { id: *sid, gpu_id: *gpu_id, max_mb: *max_mb })
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    })
                }
                ComputeDevice::Cpu { .. } => {
                    // Для CPU используем RAM
                    device_plan.storage_devices.iter().find_map(|s| {
                        if let StorageDevice::Ram { id, max_mb } = s {
                            Some(StorageDevice::Ram { id: *id, max_mb: *max_mb })
                        } else {
                            None
                        }
                    })
                }
            }
        } else {
            None
        };

        // Пытаемся зарезервировать память
        let reserve_result = memory_executor.reserve_memory(mem_kind, memory_need.total_elements);
        match reserve_result {
            Ok(()) => {
                // Успешно зарезервировали – возвращаем это устройство
                return Ok((compute_device, param_storage));
            }
            Err(MemoryError::OutOfMemory(_)) => {
                // Не хватает памяти – пробуем следующее устройство
                continue;
            }
            Err(e) => {
                // Другая ошибка – логируем и пробуем следующее
                eprintln!("[assign_devices] Ошибка резервирования памяти: {:?}", e);
                continue;
            }
        }
    }

    // Если ни одно устройство не подошло – возвращаем ошибку
    Err(MemoryError::OutOfMemory(MemoryDeviceKind::HostRam))
}

/// Автоматически распределяет устройства для всех сегментов модели,
/// учитывая ограничения по памяти (RAM/VRAM).
///
/// # Аргументы
/// * `segments` – список сегментов модели.
/// * `device_plan` – план устройств (вычислительные и хранилища).
/// * `memory_executor` – исполнитель памяти для резервирования.
/// * `batch_size` – размер батча (для оценки активаций).
///
/// # Возвращает
/// * `Ok(Vec<SegmentPlacement>)` – размещение для каждого сегмента.
/// * `Err(String)` – если не хватает памяти или другая ошибка.
pub fn assign_devices(
    segments: &[Segment],
    device_plan: &DevicePlan,
    memory_executor: &mut MemoryExecutor,
    batch_size: usize,
) -> Result<Vec<SegmentPlacement>, String> {
    let default_cpu = device_plan
        .compute_devices
        .iter()
        .find_map(|d| {
            if let ComputeDevice::Cpu { id, threads } = d {
                Some(ComputeDevice::Cpu { id: *id, threads: *threads })
            } else {
                None
            }
        })
        .unwrap_or(ComputeDevice::Cpu { id: 0, threads: 1 });
    let first_gpu = device_plan.compute_devices.iter().find_map(|d| {
        if let ComputeDevice::Gpu { id } = d {
            Some(ComputeDevice::Gpu { id: *id })
        } else {
            None
        }
    });

    let mut placements = Vec::with_capacity(segments.len());
    let mut last_compute = if let Some(ref gpu) = first_gpu {
        gpu.clone()
    } else {
        default_cpu.clone()
    };

    for (idx, segment) in segments.iter().enumerate() {
        // Вычисляем потребность в памяти
        let memory_need = calculate_segment_memory(segment, batch_size);

        // Пытаемся назначить устройство
        let (compute_device, param_storage) = match segment {
            Segment::UniversalProcessor(_layers, _, _) => {
                // Используем общую логику
                try_assign_device(segment, memory_need, device_plan, memory_executor, &last_compute)
                    .map_err(|e| format!("Не удалось назначить устройство для сегмента {}: {:?}", idx, e))?
            }
            Segment::SplitterConnector { .. }
            | Segment::CombinerConnector { .. }
            | Segment::Splitter { .. }
            | Segment::Combiner { .. } => {
                // Для коннекторов и сплиттеров стараемся держаться на том же устройстве, что и предыдущий
                // Пытаемся зарезервировать на last_compute, если не получится – fallback
                let preferred = last_compute.clone();
                // Проверяем, можем ли зарезервировать на preferred
                let mem_kind = match &preferred {
                    ComputeDevice::Cpu { .. } => MemoryDeviceKind::HostRam,
                    ComputeDevice::Gpu { id } => MemoryDeviceKind::DeviceVram(DeviceId(*id)),
                };
                match memory_executor.reserve_memory(mem_kind, memory_need.total_elements) {
                    Ok(()) => {
                        // Успешно на preferred
                        (preferred, None)
                    }
                    Err(_) => {
                        // Fallback на другие устройства
                        try_assign_device(segment, memory_need, device_plan, memory_executor, &last_compute)
                            .map_err(|e| format!("Не удалось назначить устройство для сегмента {}: {:?}", idx, e))?
                    }
                }
            }
            Segment::Unsqueeze(_) | Segment::ReduceMean(_) => {
                // Операции изменения размерности – обычно на том же устройстве, что и предыдущий
                let preferred = last_compute.clone();
                let mem_kind = match &preferred {
                    ComputeDevice::Cpu { .. } => MemoryDeviceKind::HostRam,
                    ComputeDevice::Gpu { id } => MemoryDeviceKind::DeviceVram(DeviceId(*id)),
                };
                match memory_executor.reserve_memory(mem_kind, memory_need.total_elements) {
                    Ok(()) => (preferred, None),
                    Err(_) => {
                        // Fallback на другие устройства
                        try_assign_device(segment, memory_need, device_plan, memory_executor, &last_compute)
                            .map_err(|e| format!("Не удалось назначить устройство для сегмента {}: {:?}", idx, e))?
                    }
                }
            }
        };

        // Сохраняем размещение
        placements.push(SegmentPlacement {
            segment_index: idx,
            compute_device: compute_device.clone(),
            parameter_storage: param_storage,
        });

        // Обновляем last_compute для следующих сегментов (если они без параметров)
        last_compute = compute_device;
    }

    Ok(placements)
}

/// Анализирует список слоёв UniversalProcessor и возвращает (use_gpu, has_params).
/// use_gpu = true, если GPU доступен и хотя бы один слой содержит много параметров.
/// has_params = true, если есть обучаемые параметры.
fn analyze_layers(layers: &[Box<dyn UniversalLayer>], has_gpu: bool, gpu_available: bool) -> (bool, bool) {
    if !gpu_available || !has_gpu {
        return (false, layers.iter().any(|l| l.param_len() > 0));
    }
    let large_param_threshold = 1000; // число параметров – эвристика
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