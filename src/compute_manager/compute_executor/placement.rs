// src/compute_manager/compute_executor/placement.rs

use crate::compute_manager::graph::types::Model;
use crate::device_plan::plan::{ComputeDevice, DevicePlan, StorageDevice};

/// Информация о размещении одной модели вычислительного графа.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelPlacement {
    /// Индекс модели в списке models.
    pub model_index: usize,
    /// На каком вычислительном устройстве исполняется модель.
    pub compute_device: ComputeDevice,
    /// Где хранятся параметры этой модели (если есть).
    /// None — если модель не имеет обучаемых параметров
    /// (например, ReLU, Sigmoid, Unsqueeze, коннекторы).
    pub parameter_storage: Option<StorageDevice>,
}

/// Выполняет начальное назначение вычислительных устройств для моделей
/// без учёта текущей загрузки памяти (без резервирования).
///
/// # Аргументы
/// * `models` – список моделей вычислительного графа.
/// * `device_plan` – план устройств (вычислительные и хранилища).
/// * `batch_size` – размер батча (пока используется только для информации, не влияет на решение).
///
/// # Возвращает
/// Вектор `ModelPlacement` с назначенными устройствами.
pub fn assign_initial(
    models: &[Model],
    device_plan: &DevicePlan,
    _batch_size: usize,
) -> Vec<ModelPlacement> {
    // Если нет ни одного устройства, возвращаем пустой вектор.
    if device_plan.compute_devices.is_empty() {
        return Vec::new();
    }

    // Определяем эталонные CPU и GPU (первые попавшиеся)
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

    let gpu_device = device_plan
        .compute_devices
        .iter()
        .find_map(|d| {
            if let ComputeDevice::Gpu { id } = d {
                Some(ComputeDevice::Gpu { id: *id })
            } else {
                None
            }
        });

    let mut placements = Vec::with_capacity(models.len());
    let mut last_compute = gpu_device.unwrap_or_else(|| default_cpu.clone());

    for (idx, model) in models.iter().enumerate() {
        // Выбираем устройство на основе эвристики «тяжести» модели.
        let compute_device = pick_initial_device(model, &device_plan.compute_devices, &last_compute);

        // Определяем необходимость и место хранения параметров.
        let has_params = model_has_params(model);
        let parameter_storage = if has_params {
            match &compute_device {
                ComputeDevice::Gpu { id } => {
                    // Ищем VRAM, привязанную к этому GPU
                    device_plan.storage_devices.iter().find_map(|s| {
                        if let StorageDevice::Vram { gpu_id, id: sid, max_mb } = s {
                            if *gpu_id == *id {
                                Some(StorageDevice::Vram {
                                    id: *sid,
                                    gpu_id: *gpu_id,
                                    max_mb: *max_mb,
                                })
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
        } else {
            None
        };

        placements.push(ModelPlacement {
            model_index: idx,
            compute_device: compute_device.clone(),
            parameter_storage,
        });

        last_compute = compute_device;
    }

    // Оптимизация: коннекторы и операции размерности оставляем на том же устройстве,
    // что и соседние модели, чтобы избежать лишних копирований.
    optimize_connectors(models, &mut placements);

    placements
}

/// Простейшая эвристика выбора устройства для модели.
fn pick_initial_device(
    model: &Model,
    devices: &[ComputeDevice],
    last_device: &ComputeDevice,
) -> ComputeDevice {
    let has_gpu = devices.iter().any(|d| matches!(d, ComputeDevice::Gpu { .. }));
    if !has_gpu {
        // Если GPU отсутствует, возвращаем первый CPU (или last_device, если это CPU)
        return devices
            .iter()
            .find_map(|d| {
                if let ComputeDevice::Cpu { id, threads } = d {
                    Some(ComputeDevice::Cpu { id: *id, threads: *threads })
                } else {
                    None
                }
            })
            .unwrap_or_else(|| last_device.clone());
    }

    // Оцениваем «тяжесть» модели.
    let heavy = match model {
        Model::UniversalProcessor(layers, _slices, _) => {
            let total_params: usize = layers.iter().map(|l| l.param_len()).sum();
            // Порог: если параметров больше 1000, то GPU предпочтительнее.
            total_params > 1000
        }
        Model::Splitter { .. } | Model::Combiner { .. } => true,
        _ => false,
    };

    if heavy {
        // Берём первый доступный GPU.
        devices
            .iter()
            .find_map(|d| {
                if let ComputeDevice::Gpu { id } = d {
                    Some(ComputeDevice::Gpu { id: *id })
                } else {
                    None
                }
            })
            .unwrap_or_else(|| last_device.clone())
    } else {
        // Оставляем на CPU (первый попавшийся).
        devices
            .iter()
            .find_map(|d| {
                if let ComputeDevice::Cpu { id, threads } = d {
                    Some(ComputeDevice::Cpu { id: *id, threads: *threads })
                } else {
                    None
                }
            })
            .unwrap_or_else(|| last_device.clone())
    }
}

/// Проверяет, есть ли у модели обучаемые параметры.
fn model_has_params(model: &Model) -> bool {
    match model {
        Model::UniversalProcessor(layers, _slices, _) => layers.iter().any(|l| l.param_len() > 0),
        Model::Splitter { .. } | Model::Combiner { .. } => true,
        _ => false,
    }
}

/// Подстраивает размещение коннекторов и операций размерности так,
/// чтобы они использовали то же устройство, что и соседняя вычислительная модель.
pub(crate) fn optimize_connectors(
    models: &[Model],
    placements: &mut Vec<ModelPlacement>,
) {
    let n = models.len();
    if n == 0 {
        return;
    }

    // Собираем текущие назначенные устройства.
    let mut devices: Vec<ComputeDevice> = placements.iter().map(|p| p.compute_device.clone()).collect();

    // Проход слева направо: коннекторы и dim‑операции привязываются к левому соседу.
    for i in 1..n {
        if is_connector_or_dimop(&models[i]) {
            devices[i] = devices[i - 1].clone();
        }
    }

    // Проход справа налево: подстраиваем те, что не были охвачены левым проходом.
    for i in (0..n - 1).rev() {
        if is_connector_or_dimop(&models[i]) {
            devices[i] = devices[i + 1].clone();
        }
    }

    // Применяем изменения.
    for i in 0..n {
        placements[i].compute_device = devices[i].clone();
    }
}

/// Является ли модель коннектором или операцией изменения размерности.
fn is_connector_or_dimop(model: &Model) -> bool {
    matches!(
        model,
        Model::SplitterConnector { .. }
            | Model::CombinerConnector { .. }
            | Model::Unsqueeze(_)
            | Model::ReduceMean(_)
    )
}