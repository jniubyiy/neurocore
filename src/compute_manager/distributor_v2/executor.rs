// src/compute_manager/distributor_v2/executor.rs
//
// SmartDistributor — центральный распределитель заданий v2.
//
// Знает:
//   * три оператора (Memory / CPU / GPU);
//   * TopologySnapshot (текущее состояние ресурсов);
//   * DistributionPlan (текущий план размещения);
//   * TempMatrixPool (общий для графа и CPU-оператора);
//   * счётчик эпох.
//
// Делает:
//   * dispatch(job) -> JobResult;
//   * dispatch_batch(jobs) -> Vec<JobResult>;
//   * ensure_local(handle, target);
//   * ensure_inputs_local(job, op, snapshot) — приватный pre-flight:
//     приводит входные буферы job'а (а для ConnectorOp — и saved) к
//     тому месту памяти, где будет работать выбранный оператор.
//     Реализуется через ensure_local (то есть через MemoryOperatorV2),
//     потому что распределитель — единственное место, которое знает
//     про всех трёх операторов.
//   * on_epoch_boundary(segments);
//   * prepare_job (вкладывает gpu_compute в OptimizerStep).
//
// Не знает:
//   * про потоки, scheduler, mini-model;
//   * про граф, слои, лоссы, оптимизаторы.

use std::sync::{Arc, Mutex};

use crate::compute_manager::compute_executor::ComputeExecutor;
use crate::compute_manager::cpu::cost::CostModel;
use crate::compute_manager::cpu::hardware::CPU_INFO;
use crate::compute_manager::cpu::scheduler::Scheduler;
use crate::compute_manager::cpu::{ComputeThreadPool, ControlThreadPool};
use crate::compute_manager::gpu::GpuCompute;
use crate::compute_manager::jobs_v2::{
    Job, JobHandle, JobResult, MigrateJob, OperatorKind,
};
use crate::compute_manager::matrix_buffer::{MatrixBufferHandle, TempMatrixPool};
use crate::compute_manager::memory_executor::types::MemoryDeviceKind;
use crate::compute_manager::operators_v2::{
    CpuOperatorV2, GpuOperatorV2, MemoryOperatorV2, OperatorV2,
};
use crate::device_plan::DevicePlan;

use super::plan::{DistributionPlan, SegmentTopologyInfo};
use super::strategy;
use super::topology::TopologySnapshot;

// ============================================================================
// SmartDistributor
// ============================================================================

/// Распределитель заданий v2.
pub struct SmartDistributor {
    memory_op: Arc<MemoryOperatorV2>,
    cpu_op: Arc<CpuOperatorV2>,
    gpu_op: Option<Arc<GpuOperatorV2>>,

    /// Снимок топологии.
    snapshot: Mutex<TopologySnapshot>,

    /// Текущий план размещения.
    plan: Mutex<DistributionPlan>,

    /// Общий TempMatrixPool: используется и CpuOperatorV2, и GraphV2
    /// (для конвертаций DynamicTensor ↔ Handle).
    temp_pool: Arc<Mutex<TempMatrixPool>>,

    /// Счётчик эпох.
    epoch_counter: Mutex<usize>,
}

impl SmartDistributor {
    /// Создаёт распределитель на основе `DevicePlan`.
    pub fn new(device_plan: &DevicePlan) -> Result<Self, String> {
        // 1. MemoryExecutor
        let (memory_executor, _gpu_ctx) = device_plan.build_memory_executor();

        // 2. ComputeExecutor — только чтобы получить GpuCompute
        let compute_executor = Arc::new(
            ComputeExecutor::new(device_plan.clone(), memory_executor.clone())
                .map_err(|e| format!("SmartDistributor::new: ComputeExecutor: {}", e))?,
        );
        let gpu_compute: Option<Arc<GpuCompute>> = compute_executor.gpu_compute();

        let has_gpu = gpu_compute.is_some();
        let gpu_device_id = gpu_compute.as_ref().map(|gc| gc.gpu_device_id.0);

        // 3. Потоки
        let total_cpu_threads = compute_executor.cpu_threads();
        let (control_threads, compute_threads) =
            split_cpu_threads(total_cpu_threads, has_gpu);

        let cost = CostModel::calibrate();
        let scheduler = Arc::new(Mutex::new(Scheduler::new_with_cpus(
            cost,
            CPU_INFO.clone(),
            compute_threads,
            has_gpu,
        )));
        scheduler
            .lock()
            .unwrap()
            .set_num_workers(compute_threads);

        let compute_pool = ComputeThreadPool::new(compute_threads, scheduler.clone());
        let control_pool = ControlThreadPool::new(control_threads);

        // 4. TempMatrixPool
        let temp_pool = Arc::new(Mutex::new(TempMatrixPool::new(memory_executor.clone())));

        // 5. Операторы
        let memory_op = Arc::new(MemoryOperatorV2::new(memory_executor.clone()));
        let cpu_op = Arc::new(CpuOperatorV2::new(
            scheduler,
            compute_pool,
            control_pool,
            temp_pool.clone(),
            memory_executor.clone(),
        ));
        let gpu_op = gpu_compute
            .as_ref()
            .map(|gc| Arc::new(GpuOperatorV2::new(gc.clone(), memory_executor.clone())));

        // 6. Снимок и пустой план
        let snapshot = TopologySnapshot {
            has_gpu,
            gpu_device_id,
            cpu_workers: compute_threads,
            memory_executor: memory_executor.clone(),
            gpu_compute,
            epoch: 0,
        };
        let initial_plan = DistributionPlan {
            epoch: 0,
            placements: Vec::new(),
        };

        Ok(Self {
            memory_op,
            cpu_op,
            gpu_op,
            snapshot: Mutex::new(snapshot),
            plan: Mutex::new(initial_plan),
            temp_pool,
            epoch_counter: Mutex::new(0),
        })
    }

    /// Общий TempMatrixPool. Используется графом для конвертаций
    /// `DynamicTensor ↔ MatrixBufferHandle`.
    pub fn temp_pool(&self) -> Arc<Mutex<TempMatrixPool>> {
        self.temp_pool.clone()
    }

    // -----------------------------------------------------------------------
    // dispatch
    // -----------------------------------------------------------------------

    pub fn dispatch(&self, job: Job) -> JobResult {
        let snapshot = self.snapshot.lock().unwrap().clone();
        let job = self.prepare_job(job, &snapshot);
        let op = strategy::select_operator(&job, &snapshot);

        // Pre-flight: приводим входные буферы job'а к месту памяти
        // выбранного оператора.
        let job = match self.ensure_inputs_local(job, op, &snapshot) {
            Ok(j) => j,
            Err(msg) => return JobResult::Failed(msg),
        };

        let handle = match op {
            OperatorKind::Memory => self.memory_op.submit(job),
            OperatorKind::Cpu => self.cpu_op.submit(job),
            OperatorKind::Gpu => match &self.gpu_op {
                Some(g) => g.submit(job),
                None => {
                    return JobResult::Failed(
                        "SmartDistributor::dispatch: strategy chose GPU, but gpu_op is None"
                            .into(),
                    );
                }
            },
        };

        match op {
            OperatorKind::Memory => self.memory_op.drain(handle),
            OperatorKind::Cpu => self.cpu_op.drain(handle),
            OperatorKind::Gpu => self
                .gpu_op
                .as_ref()
                .expect("gpu_op must be Some here")
                .drain(handle),
        }
    }

    pub fn dispatch_batch(&self, jobs: Vec<Job>) -> Vec<JobResult> {
        let snapshot = self.snapshot.lock().unwrap().clone();
        let mut handle_list: Vec<(OperatorKind, JobHandle)> =
            Vec::with_capacity(jobs.len());
        let mut results: Vec<Option<JobResult>> =
            (0..jobs.len()).map(|_| None).collect();

        for (idx, job) in jobs.into_iter().enumerate() {
            let job = self.prepare_job(job, &snapshot);
            let op = strategy::select_operator(&job, &snapshot);

            let job = match self.ensure_inputs_local(job, op, &snapshot) {
                Ok(j) => j,
                Err(msg) => {
                    results[idx] = Some(JobResult::Failed(msg));
                    continue;
                }
            };

            let handle = match op {
                OperatorKind::Memory => self.memory_op.submit(job),
                OperatorKind::Cpu => self.cpu_op.submit(job),
                OperatorKind::Gpu => match &self.gpu_op {
                    Some(g) => g.submit(job),
                    None => {
                        results[idx] = Some(JobResult::Failed(
                            "SmartDistributor::dispatch_batch: strategy chose GPU, \
                             but gpu_op is None"
                                .into(),
                        ));
                        continue;
                    }
                },
            };
            handle_list.push((op, handle));
        }

        for (op, handle) in handle_list {
            let res = match op {
                OperatorKind::Memory => self.memory_op.drain(handle),
                OperatorKind::Cpu => self.cpu_op.drain(handle),
                OperatorKind::Gpu => self
                    .gpu_op
                    .as_ref()
                    .expect("gpu_op must be Some here")
                    .drain(handle),
            };
            if let Some(slot) = results.iter_mut().find(|s| s.is_none()) {
                *slot = Some(res);
            }
        }

        results
            .into_iter()
            .map(|r| {
                r.unwrap_or_else(|| {
                    JobResult::Failed(
                        "SmartDistributor::dispatch_batch: missing result".into(),
                    )
                })
            })
            .collect()
    }

    // -----------------------------------------------------------------------
    // ensure_local
    // -----------------------------------------------------------------------

    pub fn ensure_local(
        &self,
        handle: MatrixBufferHandle,
        target: MemoryDeviceKind,
    ) -> Result<MatrixBufferHandle, String> {
        if handle.device_kind() == target {
            return Ok(handle);
        }
        let job = Job::Migrate(MigrateJob {
            handle: handle.clone(),
            target,
        });
        match self.dispatch(job) {
            JobResult::Unit => Ok(handle),
            JobResult::Failed(msg) => Err(msg),
            _ => Err("SmartDistributor::ensure_local: unexpected JobResult variant".into()),
        }
    }

    // -----------------------------------------------------------------------
    // on_epoch_boundary
    // -----------------------------------------------------------------------

    /// Пересчитывает план размещения и выполняет миграции.
    ///
    /// Мигрируются **все три** буфера сегмента: `params`, `grads`,
    /// `opt_state` (если он есть). Раньше мигрировались только `params`,
    /// из-за чего при работе на GPU `grads` оставались на CPU и следующий
    /// backward падал на assert `grad_params_handle.is_gpu()`.
    pub fn on_epoch_boundary(
        &self,
        segments: &[SegmentTopologyInfo],
    ) -> Result<(), String> {
        let new_plan = {
            let mut snap = self.snapshot.lock().unwrap();
            snap.epoch = snap.epoch.wrapping_add(1);
            DistributionPlan::build(&snap, segments)
        };

        for (idx, seg) in segments.iter().enumerate() {
            let target = new_plan
                .placement_for(idx)
                .map(|p| p.storage)
                .unwrap_or(MemoryDeviceKind::HostRam);

            // Мигрируем каждый из трёх буферов сегмента (если он есть).
            let buf_list: [(&str, &Option<MatrixBufferHandle>); 3] = [
                ("params", &seg.param_handle),
                ("grads", &seg.grad_handle),
                ("opt_state", &seg.opt_state_handle),
            ];

            for (label, handle_opt) in buf_list {
                let Some(handle) = handle_opt else {
                    continue;
                };
                if handle.device_kind() == target {
                    continue;
                }

                let job = Job::Migrate(MigrateJob {
                    handle: handle.clone(),
                    target,
                });
                match self.dispatch(job) {
                    JobResult::Unit => {}
                    JobResult::Failed(msg) => {
                        return Err(format!(
                            "SmartDistributor::on_epoch_boundary: seg {} {} migrate: {}",
                            idx, label, msg
                        ));
                    }
                    _ => {
                        return Err(format!(
                            "SmartDistributor::on_epoch_boundary: seg {} {} unexpected JobResult",
                            idx, label
                        ));
                    }
                }
            }
        }

        *self.plan.lock().unwrap() = new_plan;
        *self.epoch_counter.lock().unwrap() += 1;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Аксессоры
    // -----------------------------------------------------------------------

    pub fn snapshot(&self) -> TopologySnapshot {
        self.snapshot.lock().unwrap().clone()
    }

    pub fn plan(&self) -> DistributionPlan {
        self.plan.lock().unwrap().clone()
    }

    pub fn gpu_compute(&self) -> Option<Arc<GpuCompute>> {
        self.snapshot.lock().unwrap().gpu_compute.clone()
    }

    // -----------------------------------------------------------------------
    // Внутреннее
    // -----------------------------------------------------------------------

    fn prepare_job(&self, mut job: Job, snapshot: &TopologySnapshot) -> Job {
        if let Job::OptimizerStep(ref mut o) = job {
            if (o.params.is_gpu() || o.grads.is_gpu()) && o.gpu_compute.is_none() {
                o.gpu_compute = snapshot.gpu_compute.clone();
            }
        }
        job
    }

    /// Pre-flight миграция входных буферов job'а.
    ///
    /// Для ForwardSegment / BackwardSegment / Loss / ConnectorOp —
    /// целевое место памяти определяется выбранным оператором
    /// (HostRam для CPU, VRAM для GPU). Для ConnectorOp мигрируются
    /// также буферы `saved` (forward-состояние).
    ///
    /// Для DimOp — всегда HostRam, потому что реализация DimOp
    /// в `CpuOperatorV2` работает только с CPU-буферами.
    ///
    /// Для OptimizerStep миграции не выполняются: гибридный шаг
    /// (`OptimizerExpr::step_buffered_handle_hybrid`) сам решает, как
    /// достать буферы.
    ///
    /// Для Migrate и ParamInit миграции не нужны.
    fn ensure_inputs_local(
        &self,
        mut job: Job,
        op: OperatorKind,
        snapshot: &TopologySnapshot,
    ) -> Result<Job, String> {
        // Migrate — это уже миграция; ParamInit — CPU-only.
        if matches!(job, Job::Migrate(_) | Job::ParamInit(_)) {
            return Ok(job);
        }

        // OptimizerStep сам решает, как достать свои буферы.
        if matches!(job, Job::OptimizerStep(_)) {
            return Ok(job);
        }

        // Для DimOp цель всегда HostRam.
        if let Job::DimOp(ref mut d) = job {
            if d.input.device_kind() != MemoryDeviceKind::HostRam {
                d.input = self
                    .ensure_local(d.input.clone(), MemoryDeviceKind::HostRam)
                    .map_err(|e| {
                        format!("SmartDistributor::ensure_inputs_local: DimOp: {}", e)
                    })?;
            }
            return Ok(job);
        }

        let target = snapshot
            .memory_kind_for_operator(op)
            .ok_or_else(|| {
                format!(
                    "SmartDistributor::ensure_inputs_local: no memory kind for operator {:?}",
                    op
                )
            })?;

        match &mut job {
            Job::ForwardSegment(f) => {
                if f.input.device_kind() != target {
                    f.input = self.ensure_local(f.input.clone(), target).map_err(|e| {
                        format!(
                            "SmartDistributor::ensure_inputs_local: ForwardSegment seg {}: {}",
                            f.segment_index, e
                        )
                    })?;
                }
            }
            Job::BackwardSegment(b) => {
                if b.grad_output.device_kind() != target {
                    b.grad_output =
                        self.ensure_local(b.grad_output.clone(), target).map_err(|e| {
                            format!(
                                "SmartDistributor::ensure_inputs_local: BackwardSegment seg {}: {}",
                                b.segment_index, e
                            )
                        })?;
                }
            }
            Job::Loss(l) => {
                if l.pred.device_kind() != target {
                    l.pred = self.ensure_local(l.pred.clone(), target).map_err(|e| {
                        format!("SmartDistributor::ensure_inputs_local: Loss.pred: {}", e)
                    })?;
                }
                if l.target.device_kind() != target {
                    l.target = self.ensure_local(l.target.clone(), target).map_err(|e| {
                        format!("SmartDistributor::ensure_inputs_local: Loss.target: {}", e)
                    })?;
                }
            }
            Job::ConnectorOp(c) => {
                for (i, inp) in c.inputs.iter_mut().enumerate() {
                    if inp.device_kind() != target {
                        *inp = self.ensure_local(inp.clone(), target).map_err(|e| {
                            format!(
                                "SmartDistributor::ensure_inputs_local: ConnectorOp input {}: {}",
                                i, e
                            )
                        })?;
                    }
                }
                for (i, s) in c.saved.iter_mut().enumerate() {
                    if s.device_kind() != target {
                        *s = self.ensure_local(s.clone(), target).map_err(|e| {
                            format!(
                                "SmartDistributor::ensure_inputs_local: ConnectorOp saved {}: {}",
                                i, e
                            )
                        })?;
                    }
                }
            }
            // Уже обработаны выше или не требуют миграции.
            Job::DimOp(_)
            | Job::OptimizerStep(_)
            | Job::Migrate(_)
            | Job::ParamInit(_) => {}
        }

        Ok(job)
    }
}

// ============================================================================
// Разделение CPU-потоков
// ============================================================================

fn split_cpu_threads(total_threads: usize, has_gpu: bool) -> (usize, usize) {
    assert!(
        total_threads >= 2,
        "SmartDistributor::split_cpu_threads: total CPU threads must be >= 2"
    );

    let physical_cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(total_threads);

    let mut control = ((physical_cores / 8).max(1)).min(4);
    control = control.min(total_threads - 1);

    let mut compute = total_threads - control;
    if has_gpu {
        compute = compute.min((physical_cores / 2).max(1));
    }
    compute = compute.max(1).min(total_threads - control);

    (control, compute)
}