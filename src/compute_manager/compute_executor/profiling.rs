// src/compute_manager/compute_executor/profiling.rs

use std::collections::HashMap;

use crate::device_plan::plan::ComputeDevice;

use super::DeviceHashKey;

/// Накопленная профилировочная статистика для адаптивного планирования.
#[derive(Clone)]
pub struct ProfilingState {
    /// Для каждого сегмента и устройства хранится пара
    /// (суммарное время в наносекундах, количество измерений).
    segment_timings: HashMap<(usize, DeviceHashKey), (f64, usize)>,
    /// Номер последней эпохи, на которой выполнялось перераспределение.
    last_reassign_epoch: usize,
}

impl ProfilingState {
    /// Создаёт пустой профиль.
    pub fn new() -> Self {
        Self {
            segment_timings: HashMap::new(),
            last_reassign_epoch: 0,
        }
    }

    /// Записывает время выполнения сегмента на заданном устройстве.
    pub fn add_timing(&mut self, seg_index: usize, device: ComputeDevice, duration_ns: f64) {
        let key = DeviceHashKey::from(&device);
        let entry = self
            .segment_timings
            .entry((seg_index, key))
            .or_insert((0.0, 0));
        entry.0 += duration_ns;
        entry.1 += 1;
    }

    /// Возвращает среднее время выполнения сегмента на устройстве (в наносекундах),
    /// если данные есть.
    pub fn average_time(&self, seg_index: usize, device: &ComputeDevice) -> Option<f64> {
        let key = DeviceHashKey::from(device);
        self.segment_timings
            .get(&(seg_index, key))
            .map(|(total, count)| total / *count as f64)
    }

    /// Проверяет, нужно ли выполнить перераспределение на основе текущего номера эпохи.
    ///
    /// Возвращает `true`, если с момента последнего перераспределения прошла хотя бы одна эпоха.
    pub fn should_reassign(&self, current_epoch: usize) -> bool {
        current_epoch > self.last_reassign_epoch
    }

    /// Отмечает, что перераспределение было выполнено на указанной эпохе.
    pub fn mark_reassigned(&mut self, epoch: usize) {
        self.last_reassign_epoch = epoch;
    }

    /// Возвращает все накопленные тайминги (для отладки или анализа).
    pub fn timings(&self) -> &HashMap<(usize, DeviceHashKey), (f64, usize)> {
        &self.segment_timings
    }

    /// Очищает накопленную статистику (например, при смене конфигурации модели).
    pub fn clear(&mut self) {
        self.segment_timings.clear();
        self.last_reassign_epoch = 0;
    }
}