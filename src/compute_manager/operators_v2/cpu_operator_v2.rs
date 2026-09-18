// src/compute_manager/operators_v2/cpu_operator_v2.rs
//
// CpuOperatorV2 — «мир CPU».
//
// Знает:
//   * ControlThreadPool, ComputeThreadPool;
//   * Scheduler (вместе с mini-model'ями);
//   * TempMatrixPool;
//   * Arc<RwLock<MemoryExecutor>>;
//   * Arc<Mutex<HashMap<usize, OptimizerExpr>>> (состояния оптимизаторов).
//
// Делает:
//   * submit(job) -> JobHandle — синхронно исполняет job и кладёт результат;
//   * drain(handle) -> JobResult;
//   * capacity() -> OperatorCapacity.
//
// Не знает:
//   * какой сегмент, какой loss пришёл;
//   * что за граф его дёргает.
//
// Про синхронность submit: см. комментарий ниже.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, RwLock};

use crate::compute_manager::operators_v2::cpu_v2::{ComputeThreadPool, ControlThreadPool, Scheduler};
use crate::compute_manager::jobs_v2::{
    Job, JobHandle, JobResult, OperatorKind, ParamInitJob,
};
use crate::compute_manager::operators_v2::memory_v2::buffer::TempMatrixPool;
use crate::compute_manager::operators_v2::memory_v2::MemoryExecutor;
use crate::optimizer_plan::OptimizerExpr;

use super::capacity_v2::OperatorCapacity;
use super::cpu_v2::chunk_dispatch_v2;
use super::cpu_v2::loss_dispatch_v2;
use super::cpu_v2::opt_dispatch_v2;
use super::cpu_v2::scheduler_v2::SchedulerV2;
use super::operator_v2::OperatorV2;

// ============================================================================
// Внутреннее состояние
// ============================================================================

struct CpuOperatorInner {
    kind: OperatorKind,
    scheduler: SchedulerV2,
    compute_pool: ComputeThreadPool,
    #[allow(dead_code)]
    control_pool: ControlThreadPool,
    pool: Arc<Mutex<TempMatrixPool>>,
    memory_executor: Arc<RwLock<MemoryExecutor>>,
    optimizers: Arc<Mutex<HashMap<usize, OptimizerExpr>>>,
    slots: Arc<Mutex<HashMap<u64, JobResult>>>,
    cv: Arc<Condvar>,
    next_id: AtomicU64,
}

impl CpuOperatorInner {
    fn execute(&self, job: Job) -> (JobResult, crate::compute_manager::jobs_v2::JobKind) {
        use crate::compute_manager::jobs_v2::JobKind;
        let kind = job.kind();

        let result = match job {
            Job::ForwardSegment(f) => chunk_dispatch_v2::execute_forward(
                f,
                &self.compute_pool,
                Arc::clone(&self.pool),
            ),
            Job::BackwardSegment(b) => chunk_dispatch_v2::execute_backward(
                b,
                &self.compute_pool,
                Arc::clone(&self.pool),
            ),
            Job::DimOp(d) => {
                chunk_dispatch_v2::execute_dimop(d, Arc::clone(&self.pool))
            }
            Job::ConnectorOp(c) => {
                chunk_dispatch_v2::execute_connector(c, Arc::clone(&self.pool))
            }
            Job::Loss(l) => loss_dispatch_v2::execute_loss(l, Arc::clone(&self.pool)),
            Job::OptimizerStep(o) => opt_dispatch_v2::execute_optimizer_step(
                o,
                Arc::clone(&self.memory_executor),
                Arc::clone(&self.pool),
                Arc::clone(&self.optimizers),
            ),
            Job::ParamInit(p) => self.execute_param_init(p),
            Job::Migrate(_) => JobResult::Failed(
                "CpuOperatorV2: Migrate must go to MemoryOperatorV2".into(),
            ),
        };

        (result, kind)
    }

    /// Инициализация параметров одного буфера.
    ///
    /// # Порядок применения
    ///
    /// 1. Если `job.precomputed = Some(data)` — записываем значения как есть.
    ///    Это путь, используемый `GraphV2::init_params`: он генерирует всю
    ///    последовательность одним RNG (как v1 `execute.rs`) и раздаёт
    ///    буферам её куски. Даёт точное совпадение начальных весов с v1.
    ///
    /// 2. Если `job.precomputed = None` — генерируем локально по `initializer`.
    ///    Для `RandomUniform` seed комбинируется с `buffer_idx`, чтобы каждый
    ///    буфер получил **свою** последовательность (устраняет
    ///    коррелированную инициализацию между сегментами).
    fn execute_param_init(&self, job: ParamInitJob) -> JobResult {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let n = job.params.rows() * job.params.cols();

        let data: Vec<f32> = if let Some(pre) = job.precomputed {
            if pre.len() != n {
                return JobResult::Failed(format!(
                    "CpuOperatorV2::param_init: precomputed length {} != buffer size {} \
                     (buffer_idx = {})",
                    pre.len(),
                    n,
                    job.buffer_idx
                ));
            }
            pre
        } else {
            match &job.initializer {
                crate::training_plan::Initializer::Zeros => vec![0.0; n],
                crate::training_plan::Initializer::Ones => vec![1.0; n],
                crate::training_plan::Initializer::RandomUniform { min, max } => {
                    let mut rng: Box<dyn rand::RngCore> = match job.seed {
                        Some(s) => Box::new(StdRng::seed_from_u64(
                            s.wrapping_add(job.buffer_idx as u64),
                        )),
                        None => Box::new(rand::thread_rng()),
                    };
                    (0..n).map(|_| rng.gen_range(*min..*max)).collect()
                }
            }
        };

        job.params.write_range(0, &data);
        JobResult::Unit
    }
}

// ============================================================================
// Публичный оператор
// ============================================================================

pub struct CpuOperatorV2 {
    inner: Arc<CpuOperatorInner>,
}

impl CpuOperatorV2 {
    pub fn new(
        scheduler: Arc<Mutex<Scheduler>>,
        compute_pool: ComputeThreadPool,
        control_pool: ControlThreadPool,
        pool: Arc<Mutex<TempMatrixPool>>,
        memory_executor: Arc<RwLock<MemoryExecutor>>,
    ) -> Self {
        Self {
            inner: Arc::new(CpuOperatorInner {
                kind: OperatorKind::Cpu,
                scheduler: SchedulerV2::wrap(scheduler),
                compute_pool,
                control_pool,
                pool,
                memory_executor,
                optimizers: Arc::new(Mutex::new(HashMap::new())),
                slots: Arc::new(Mutex::new(HashMap::new())),
                cv: Arc::new(Condvar::new()),
                next_id: AtomicU64::new(0),
            }),
        }
    }
}

impl OperatorV2 for CpuOperatorV2 {
    fn kind(&self) -> OperatorKind {
        self.inner.kind
    }

    /// Синхронный submit.
    ///
    /// 1. Параллельный forward/backward делается через
    ///    `forward_universal_parallel` / `backward_universal_parallel`,
    ///    которые **сами** используют пул воркеров. Если запускать их
    ///    внутри ещё одной задачи в том же пуле, `wait_all` внутри
    ///    увидит текущую задачу как активную и получит дедлок.
    ///
    /// 2. Loss, optimizer step, DimOp, ConnectorOp — короткие операции,
    ///    исполняются в вызывающем потоке.
    ///
    /// Результат кладётся в слот немедленно; `drain` возвращает
    /// `JobResult` без ожидания.
    fn submit(&self, job: Job) -> JobHandle {
        let id = self.inner.next_id.fetch_add(1, Ordering::Relaxed);
        let (result, kind) = self.inner.execute(job);

        {
            let mut slots = self.inner.slots.lock().unwrap();
            slots.insert(id, result);
        }
        self.inner.cv.notify_all();

        JobHandle {
            id,
            kind,
            operator: OperatorKind::Cpu,
        }
    }

    fn drain(&self, handle: JobHandle) -> JobResult {
        assert_eq!(
            handle.operator,
            OperatorKind::Cpu,
            "CpuOperatorV2::drain: handle belongs to {:?}",
            handle.operator
        );
        let mut slots = self.inner.slots.lock().unwrap();
        slots
            .remove(&handle.id)
            .expect("CpuOperatorV2::drain: result already taken")
    }

    fn try_drain(&self, handle: JobHandle) -> Option<JobResult> {
        let mut slots = self.inner.slots.lock().unwrap();
        slots.remove(&handle.id)
    }

    fn capacity(&self) -> OperatorCapacity {
        OperatorCapacity::idle(OperatorKind::Cpu)
    }
}