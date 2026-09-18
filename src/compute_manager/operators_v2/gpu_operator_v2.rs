// src/compute_manager/operators_v2/gpu_operator_v2.rs
//
// GpuOperatorV2 — «мир GPU».
//
// Знает:
//   * Arc<GpuCompute>;
//   * Arc<RwLock<MemoryExecutor>>;
//   * свою очередь (GpuQueueV2).
//
// Делает:
//   * submit(job) -> JobHandle — кладёт в очередь и возвращается немедленно;
//   * drain(handle) -> JobResult — ждёт на condvar, пока GPU-тред
//     положит результат;
//   * try_drain(handle) — неблокирующий опрос.
//
// Не знает:
//   * про CPU, Scheduler, mini-model, граф.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, RwLock};

use crate::compute_manager::operators_v2::gpu_v2::GpuCompute;
use crate::compute_manager::jobs_v2::{
    Job, JobHandle, JobResult, OperatorKind,
};
use crate::compute_manager::operators_v2::memory_v2::MemoryExecutor;

use super::capacity_v2::OperatorCapacity;
use super::gpu_v2::queue_v2::{GpuQueueV2, GpuTaskMessage};
use super::operator_v2::OperatorV2;

// ============================================================================
// Внутреннее устройство
// ============================================================================

struct GpuOperatorInner {
    kind: OperatorKind,

    /// Очередь + выделенный GPU-тред.
    queue: GpuQueueV2,

    /// Слоты результатов.
    slots: Arc<Mutex<HashMap<u64, JobResult>>>,

    /// Condvar для ожидания готовности результата.
    cv: Arc<Condvar>,

    /// Следующий идентификатор.
    next_id: AtomicU64,
}

// ============================================================================
// Публичный оператор
// ============================================================================

/// Оператор GPU v2.
pub struct GpuOperatorV2 {
    inner: Arc<GpuOperatorInner>,
}

impl GpuOperatorV2 {
    /// Создаёт оператор: запускает выделенный GPU-тред.
    pub fn new(
        gpu: Arc<GpuCompute>,
        memory: Arc<RwLock<MemoryExecutor>>,
    ) -> Self {
        Self {
            inner: Arc::new(GpuOperatorInner {
                kind: OperatorKind::Gpu,
                queue: GpuQueueV2::new(gpu, memory),
                slots: Arc::new(Mutex::new(HashMap::new())),
                cv: Arc::new(Condvar::new()),
                next_id: AtomicU64::new(0),
            }),
        }
    }
}

impl OperatorV2 for GpuOperatorV2 {
    fn kind(&self) -> OperatorKind {
        self.inner.kind
    }

    fn submit(&self, job: Job) -> JobHandle {
        let id = self.inner.next_id.fetch_add(1, Ordering::Relaxed);
        let kind = job.kind();

        self.inner.queue.send(GpuTaskMessage::Job {
            id,
            job,
            slots: Arc::clone(&self.inner.slots),
            cv: Arc::clone(&self.inner.cv),
        });

        JobHandle {
            id,
            kind,
            operator: OperatorKind::Gpu,
        }
    }

    fn drain(&self, handle: JobHandle) -> JobResult {
        assert_eq!(
            handle.operator,
            OperatorKind::Gpu,
            "GpuOperatorV2::drain: handle belongs to {:?}",
            handle.operator
        );

        let mut slots = self.inner.slots.lock().unwrap();
        loop {
            if let Some(result) = slots.remove(&handle.id) {
                return result;
            }
            slots = self.inner.cv.wait(slots).unwrap();
        }
    }

    fn try_drain(&self, handle: JobHandle) -> Option<JobResult> {
        let mut slots = self.inner.slots.lock().unwrap();
        slots.remove(&handle.id)
    }

    fn capacity(&self) -> OperatorCapacity {
        let queue_len = self.inner.slots.lock().unwrap().len();
        OperatorCapacity {
            kind: OperatorKind::Gpu,
            queue_len,
            max_queue_len: None,
            is_ready: true,
            estimated_latency_ns: None,
        }
    }
}