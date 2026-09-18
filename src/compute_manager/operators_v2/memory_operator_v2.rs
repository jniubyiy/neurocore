// src/compute_manager/operators_v2/memory_operator_v2.rs
//
// MemoryOperatorV2 — исполнитель миграций и аллокаций памяти.
//
// Знает только `MemoryExecutor`.
// Не знает: зачем мигрируют, кто именно, что за граф.
//
// `submit` синхронный: миграция одного буфера — быстрая операция
// (memcpy либо перекладывание в VRAM). Асинхронность здесь дала бы
// лишнюю сложность без выигрыша. Handle возвращается сразу после
// исполнения; `drain` забирает результат.
//
// Обрабатывает только `Job::Migrate`. Любой другой тип job'а —
// `JobResult::Failed`.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, RwLock};

use crate::compute_manager::jobs_v2::{
    Job, JobHandle, JobKind, JobResult, MigrateJob, OperatorKind,
};
use crate::compute_manager::memory_executor::MemoryExecutor;

use super::capacity_v2::OperatorCapacity;
use super::operator_v2::OperatorV2;

// ============================================================================
// Внутреннее устройство
// ============================================================================

struct MemoryOperatorInner {
    kind: OperatorKind,
    memory: Arc<RwLock<MemoryExecutor>>,
    slots: Arc<Mutex<HashMap<u64, JobResult>>>,
    cv: Arc<Condvar>,
    next_id: AtomicU64,
}

impl MemoryOperatorInner {
    fn execute_migrate(&self, job: MigrateJob) -> JobResult {
        let mut mem = self.memory.write().unwrap();
        match mem.move_matrix_handle(job.handle.id(), job.target) {
            Ok(()) => JobResult::Unit,
            Err(e) => JobResult::Failed(format!("MemoryOperatorV2::migrate: {:?}", e)),
        }
    }
}

// ============================================================================
// Публичный оператор
// ============================================================================

/// Оператор памяти v2.
pub struct MemoryOperatorV2 {
    inner: Arc<MemoryOperatorInner>,
}

impl MemoryOperatorV2 {
    /// Создаёт оператор поверх существующего `MemoryExecutor`.
    pub fn new(memory: Arc<RwLock<MemoryExecutor>>) -> Self {
        Self {
            inner: Arc::new(MemoryOperatorInner {
                kind: OperatorKind::Memory,
                memory,
                slots: Arc::new(Mutex::new(HashMap::new())),
                cv: Arc::new(Condvar::new()),
                next_id: AtomicU64::new(0),
            }),
        }
    }
}

impl OperatorV2 for MemoryOperatorV2 {
    fn kind(&self) -> OperatorKind {
        self.inner.kind
    }

    fn submit(&self, job: Job) -> JobHandle {
        let id = self.inner.next_id.fetch_add(1, Ordering::Relaxed);

        let result = match job {
            Job::Migrate(m) => self.inner.execute_migrate(m),
            other => JobResult::Failed(format!(
                "MemoryOperatorV2: unsupported job kind {:?}",
                other.kind()
            )),
        };

        let kind = match result {
            JobResult::Failed(_) => JobKind::Migrate, // маркер, детали — в сообщении
            _ => JobKind::Migrate,
        };

        {
            let mut slots = self.inner.slots.lock().unwrap();
            slots.insert(id, result);
        }
        self.inner.cv.notify_all();

        JobHandle {
            id,
            kind,
            operator: OperatorKind::Memory,
        }
    }

    fn drain(&self, handle: JobHandle) -> JobResult {
        assert_eq!(
            handle.operator,
            OperatorKind::Memory,
            "MemoryOperatorV2::drain: handle belongs to {:?}",
            handle.operator
        );
        let mut slots = self.inner.slots.lock().unwrap();
        slots
            .remove(&handle.id)
            .expect("MemoryOperatorV2::drain: result already taken")
    }

    fn try_drain(&self, handle: JobHandle) -> Option<JobResult> {
        let mut slots = self.inner.slots.lock().unwrap();
        slots.remove(&handle.id)
    }

    fn capacity(&self) -> OperatorCapacity {
        OperatorCapacity::idle(OperatorKind::Memory)
    }
}