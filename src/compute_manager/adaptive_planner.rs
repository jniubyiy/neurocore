// src/compute_manager/adaptive_planner.rs

use std::collections::HashMap;
use crate::compute_manager::device_assignment::SegmentPlacement;
use crate::compute_manager::graph::types::Segment;
use crate::compute_manager::memory_executor::MemoryExecutor;
use crate::device_plan::plan::{ComputeDevice, DevicePlan};

/// Порог числа шагов (прямых проходов), после которого запускается пересмотр размещения.
const REASSIGN_THRESHOLD: usize = 100;

/// Обёртка для использования ComputeDevice в качестве ключа хеш-таблицы.
#[derive(Clone, PartialEq, Eq, Hash)]
struct DeviceHashKey(ComputeDevice);

impl From<&ComputeDevice> for DeviceHashKey {
    fn from(dev: &ComputeDevice) -> Self {
        DeviceHashKey(dev.clone())
    }
}

/// Накопленная профилировочная статистика для адаптивного планирования.
#[derive(Clone)]
pub(crate) struct ProfilingData {
    /// Для каждого сегмента и устройства хранится пара (суммарное время в наносекундах, количество измерений).
    segment_timings: HashMap<(usize, DeviceHashKey), (f64, usize)>,
    /// Количество шагов (прямых проходов), прошедших с последнего перепланирования.
    pub(crate) steps_since_reassign: usize,
}

impl ProfilingData {
    /// Создаёт пустой профиль.
    pub(crate) fn new() -> Self {
        Self {
            segment_timings: HashMap::new(),
            steps_since_reassign: 0,
        }
    }

    /// Записывает время выполнения сегмента на заданном устройстве.
    pub(crate) fn add(&mut self, seg_index: usize, device: ComputeDevice, duration_ns: f64) {
        let key = DeviceHashKey::from(&device);
        let entry = self.segment_timings
            .entry((seg_index, key))
            .or_insert((0.0, 0));
        entry.0 += duration_ns;
        entry.1 += 1;
    }

    /// Увеличивает счётчик шагов и возвращает true, если пора пересмотреть размещение.
    pub(crate) fn tick_and_should_reassign(&mut self) -> bool {
        self.steps_since_reassign += 1;
        self.steps_since_reassign >= REASSIGN_THRESHOLD
    }

    /// Возвращает среднее время выполнения сегмента на устройстве (в наносекундах), если данные есть.
    pub(crate) fn average_time(&self, seg_index: usize, device: &ComputeDevice) -> Option<f64> {
        let key = DeviceHashKey::from(device);
        self.segment_timings
            .get(&(seg_index, key))
            .map(|(total, count)| total / *count as f64)
    }
}

/// Главная функция адаптивного размещения сегментов.
///
/// Принимает текущие сегменты, план устройств, исполнитель памяти (может не использоваться),
/// размер батча и накопленную статистику (может быть None при первом вызове).
/// Возвращает новый вектор `SegmentPlacement` и вектор флагов `keep_buffers` (той же длины, что и сегменты).
/// `keep_buffers[i] == true` означает, что старый persistent‑буфер для сегмента `i` можно сохранить
/// (устройство не изменилось). В противном случае буфер должен быть освобождён и создан заново.
pub(crate) fn assign_devices_adaptive(
    segments: &[Segment],
    device_plan: &DevicePlan,
    _executor: &mut MemoryExecutor,
    _batch_size: usize,
    profiling: Option<&ProfilingData>,
) -> (Vec<SegmentPlacement>, Vec<bool>) {
    let mut placements = Vec::with_capacity(segments.len());
    let mut keep_buffers = Vec::with_capacity(segments.len());

    // Если статистика отсутствует (первый вызов), используем начальную эвристику.
    let use_initial = profiling.is_none();
    let default_profiling = ProfilingData::new(); // живёт до конца функции
    let profiling = profiling.unwrap_or(&default_profiling);

    // Определяем все доступные вычислительные устройства из плана.
    let available_devices: Vec<ComputeDevice> = device_plan.compute_devices.clone();
    // Если устройств нет (чего быть не должно), возвращаем пустые векторы.
    if available_devices.is_empty() {
        return (placements, keep_buffers);
    }

    // Для хранения предыдущего размещения, если мы хотим сохранить буферы.
    // При первом вызове нет предыдущего размещения — все буферы будут созданы.
    let previous_placements: Option<&[SegmentPlacement]> = None; // будет передано позже, пока None

    for (idx, segment) in segments.iter().enumerate() {
        let best_device = if use_initial {
            // Начальное размещение: предпочитаем GPU для тяжёлых сегментов, иначе CPU.
            initial_device_heuristic(segment, &available_devices)
        } else {
            // Адаптивное размещение на основе профилировочных данных.
            pick_best_device(segment, idx, profiling, &available_devices)
        };

        // Решаем, изменилось ли устройство по сравнению с предыдущим размещением.
        let changed = match previous_placements {
            Some(prev) => prev.get(idx).map_or(true, |pp| pp.compute_device != best_device),
            None => true, // при первом вызове все буферы новые
        };

        // Параметры будут храниться там же, где и вычисления (упрощение).
        let placement = SegmentPlacement {
            segment_index: idx,
            compute_device: best_device,
            parameter_storage: None, // будет задано позже в модели
        };
        placements.push(placement);
        keep_buffers.push(!changed);
    }

    // Пост-оптимизация: коннекторы и операции размерности стараемся оставить на том же устройстве,
    // что и соседний вычислительный сегмент, чтобы избежать лишних копирований.
    // Проходим по сегментам и подстраиваем.
    optimize_connectors(segments, &mut placements, &mut keep_buffers);

    (placements, keep_buffers)
}

/// Начальная эвристика выбора устройства для сегмента.
/// GPU назначается, если сегмент содержит обучаемые параметры с общим числом выше порога,
/// либо если это Splitter/Combiner (они тяжёлые). Иначе CPU.
fn initial_device_heuristic(segment: &Segment, devices: &[ComputeDevice]) -> ComputeDevice {
    let has_gpu = devices.iter().any(|d| matches!(d, ComputeDevice::Gpu { .. }));
    if !has_gpu {
        return devices.first().cloned().unwrap_or(ComputeDevice::Cpu { id: 0, threads: 1 });
    }

    // Оцениваем "тяжесть" сегмента.
    let heavy = match segment {
        Segment::UniversalProcessor(layers, _slices, _) => {
            let total_params: usize = layers.iter().map(|l| l.param_len()).sum();
            total_params > 1000 // порог, выше которого выгоден GPU
        }
        Segment::Splitter { .. } | Segment::Combiner { .. } => true,
        _ => false,
    };

    if heavy {
        // Берём первый доступный GPU.
        devices.iter()
            .find_map(|d| if let ComputeDevice::Gpu { id } = d { Some(ComputeDevice::Gpu { id: *id }) } else { None })
            .unwrap_or_else(|| devices.first().cloned().unwrap())
    } else {
        // Оставляем на CPU.
        devices.iter()
            .find_map(|d| if let ComputeDevice::Cpu { id, threads } = d { Some(ComputeDevice::Cpu { id: *id, threads: *threads }) } else { None })
            .unwrap_or_else(|| devices.first().cloned().unwrap())
    }
}

/// Адаптивный выбор устройства на основе профилировочных данных.
/// Для каждого доступного устройства вычисляется среднее время (если есть), и выбирается с минимальным временем.
/// Если данных нет по какому-то устройству, используется пессимистичная оценка (большое время), чтобы попробовать его.
fn pick_best_device(
    _segment: &Segment,
    seg_index: usize,
    profiling: &ProfilingData,
    devices: &[ComputeDevice],
) -> ComputeDevice {
    let mut best_device = devices.first().cloned().unwrap();
    let mut best_time = f64::MAX;

    for device in devices {
        // Если есть данные по этому сегменту на этом устройстве, берём среднее, иначе ставим штрафное время,
        // чтобы система попробовала устройство, если другие перегружены.
        let time = profiling
            .average_time(seg_index, device)
            .unwrap_or(1e12); // очень большое время (условно)

        if time < best_time {
            best_time = time;
            best_device = device.clone();
        }
    }

    best_device
}

/// Подстраивает размещение коннекторов и операций размерности, чтобы они оставались на том же устройстве,
/// что и ближайший вычислительный сегмент (UniversalProcessor, Splitter, Combiner).
/// Это уменьшает количество копирований между устройствами.
fn optimize_connectors(
    segments: &[Segment],
    placements: &mut Vec<SegmentPlacement>,
    keep_buffers: &mut Vec<bool>,
) {
    let n = segments.len();
    if n == 0 {
        return;
    }

    // Собираем массив устройств, назначенных каждому сегменту.
    let mut devices: Vec<ComputeDevice> = placements.iter().map(|p| p.compute_device.clone()).collect();

    // Двигаемся слева направо: если сегмент — коннектор или операция размерности, берём устройство предыдущего сегмента.
    for i in 1..n {
        if is_connector_or_dimop(&segments[i]) {
            devices[i] = devices[i - 1].clone();
        }
    }

    // Двигаемся справа налево для тех же типов, чтобы они выровнялись по следующему сегменту,
    // если предыдущий отсутствует (например, первый сегмент — коннектор).
    for i in (0..n - 1).rev() {
        if is_connector_or_dimop(&segments[i]) {
            devices[i] = devices[i + 1].clone();
        }
    }

    // Применяем изменения и обновляем keep_buffers.
    for i in 0..n {
        if placements[i].compute_device != devices[i] {
            placements[i].compute_device = devices[i].clone(); // клонируем
            // Изменение устройства означает, что старый буфер не подходит.
            keep_buffers[i] = false;
        }
    }
}

/// Проверяет, является ли сегмент коннектором или операцией изменения размерности.
fn is_connector_or_dimop(segment: &Segment) -> bool {
    matches!(segment,
        Segment::SplitterConnector { .. } |
        Segment::CombinerConnector { .. } |
        Segment::Unsqueeze(_) |
        Segment::ReduceMean(_)
    )
}