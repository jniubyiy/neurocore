// src/compute_manager/operators_v2/cpu_v2/scheduler_v2.rs
//
// Тонкая обёртка над существующим `Scheduler`.
//
// Оригинальный `Scheduler` (в `compute_manager::cpu::scheduler`) уже
// содержит полный набор: `Vec<ForwardTimePredictor>` (по одной mini-model
// на CPU), `HardwareProfile`, `CostModel`, и умеет `plan_chunks_assignment`
// + `report_execution_time`. Обёртка нужна только для того, чтобы:
//
//   * тип v2 не тянул наружу сигнатуры старого API;
//   * в будущем точки замены (например, если v2 захочет свою политику)
//     были локализованы в одном файле.
//
// Реализация делегирует все вызовы существующему `Scheduler`.

use std::sync::{Arc, Mutex};

use crate::compute_manager::cpu::Scheduler;

/// Обёртка над `Scheduler`. Держит `Arc<Mutex<Scheduler>>` — тот же
/// объект может быть разделён с `ComputeThreadPool`.
pub struct SchedulerV2 {
    inner: Arc<Mutex<Scheduler>>,
}

impl SchedulerV2 {
    /// Заворачивает готовый scheduler.
    pub fn wrap(inner: Arc<Mutex<Scheduler>>) -> Self {
        Self { inner }
    }

    /// Доступ к внутреннему scheduler'у (для передачи в `ComputeThreadPool`).
    pub fn inner(&self) -> &Arc<Mutex<Scheduler>> {
        &self.inner
    }

    /// Планирует раскладку `total_tasks` задач по воркерам.
    pub fn plan_chunks_assignment(&self, total_tasks: usize) -> Vec<Vec<(usize, usize, usize)>> {
        self.inner.lock().unwrap().plan_chunks_assignment(total_tasks)
    }

    /// Обратная связь о времени исполнения чанка (обучает mini-model).
    pub fn report_execution_time(
        &self,
        worker_id: usize,
        task_size: usize,
        duration_ns: f64,
    ) {
        self.inner
            .lock()
            .unwrap()
            .report_execution_time(worker_id, task_size, duration_ns);
    }

    /// Оценка времени для задачи размера `task_size` на воркере `worker_id`.
    pub fn predict_time(&self, worker_id: usize, task_size: usize) -> Option<f64> {
        self.inner.lock().unwrap().predict_time(worker_id, task_size)
    }

    /// Число воркеров.
    pub fn num_workers(&self) -> usize {
        self.inner.lock().unwrap().num_workers()
    }
}