// src/compute_manager/operators_v2/cpu_v2/opt_dispatch_v2.rs
//
// Диспетчеры шага оптимизатора по одному буферу параметров.
//
// # Две фазы (MIGRATION_PLAN.md §7, инвариант I-1)
//
// Шаг оптимизатора разделён на два независимых job'а:
//
//   * `execute_optimizer_modify_grads` — модификация градиента
//     (lr, momentum, adam, clip, weight_decay). Не обновляет параметры.
//
//   * `execute_optimizer_apply_update` — обновление параметров
//     (`params -= grads`). Не модифицирует градиент.
//
// Между этими фазами в цикле обучения встраивается `adapter_pass`
// (per-layer коррекция градиента, Фаза 4 плана).
//
// Кубики оптимизатора — CPU-код (см. `plans::optimizer_plan::cube`),
// и они всегда исполняются на CPU. Если буферы физически лежат в VRAM,
// `OptimizerExpr::*_hybrid` скачивает их на CPU, шагает и заливает
// обратно (причём для `modify_grads` — только `grads`, для `apply_update`
// — только `params`; см. комментарии в `expr.rs`). Ссылка на
// `GpuCompute` приходит в job'е — сам CPU-оператор её не хранит.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};

use crate::compute_manager::jobs_v2::{
    JobResult, OptimizerApplyUpdateJob, OptimizerModifyGradsJob,
};
use crate::compute_manager::operators_v2::memory_v2::buffer::TempMatrixPool;
use crate::compute_manager::operators_v2::memory_v2::MemoryExecutor;
use crate::optimizer_plan::OptimizerExpr;

/// Тип общего хранилища `OptimizerExpr` между двумя фазами.
///
/// Ключ — `buffer_idx`, значение — `OptimizerExpr` с состояниями кубиков.
/// Создаётся в первой фазе (`modify_grads`) и переиспользуется во второй
/// (`apply_update`). Уничтожается вместе с `CpuOperatorV2`.
type OptimizerStore = Arc<Mutex<HashMap<usize, OptimizerExpr>>>;

/// Возвращает существующий `OptimizerExpr` для `buffer_idx` или создаёт
/// новый, инициализируя его состояния через `TempMatrixPool`.
fn ensure_optimizer_expr(
    buffer_idx: usize,
    num_params: usize,
    optimizer: &crate::optimizer_plan::OptimizerDesc,
    memory_executor: &Arc<RwLock<MemoryExecutor>>,
    pool: &Arc<Mutex<TempMatrixPool>>,
    optimizers: &OptimizerStore,
) {
    let mut map = optimizers.lock().unwrap();
    if !map.contains_key(&buffer_idx) {
        let chain = optimizer.build_chain();
        let mut p = pool.lock().unwrap();
        let expr = OptimizerExpr::new_buffered_handle(
            memory_executor.clone(),
            num_params,
            chain,
            &mut *p,
        );
        map.insert(buffer_idx, expr);
    }
}

/// Фаза 1: модификация градиента.
///
/// Применяет все кубики, кроме `ApplyUpdate`. Не обновляет параметры.
pub fn execute_optimizer_modify_grads(
    job: OptimizerModifyGradsJob,
    memory_executor: Arc<RwLock<MemoryExecutor>>,
    pool: Arc<Mutex<TempMatrixPool>>,
    optimizers: OptimizerStore,
) -> JobResult {
    // Создаём OptimizerExpr при первом обращении к buffer_idx.
    ensure_optimizer_expr(
        job.buffer_idx,
        job.params.rows(),
        &job.optimizer,
        &memory_executor,
        &pool,
        &optimizers,
    );

    let gpu_ref: Option<&crate::compute_manager::operators_v2::gpu_v2::GpuCompute> =
        job.gpu_compute.as_deref();

    let mut map = optimizers.lock().unwrap();
    let expr = map
        .get_mut(&job.buffer_idx)
        .expect("OptimizerExpr must exist after ensure_optimizer_expr");

    expr.modify_grads_step_buffered_handle_hybrid(&job.params, &job.grads, gpu_ref);

    JobResult::Unit
}

/// Фаза 2: обновление параметров.
///
/// Применяет только `ApplyUpdate` (`params -= grads`). Не модифицирует
/// градиент.
///
/// # Паника
///
/// Паникует, если `OptimizerExpr` для `buffer_idx` ещё не создан.
/// Это означает, что для данного буфера не была вызвана фаза
/// `modify_grads` — нарушение фазового порядка (инвариант I-1).
pub fn execute_optimizer_apply_update(
    job: OptimizerApplyUpdateJob,
    _memory_executor: Arc<RwLock<MemoryExecutor>>,
    _pool: Arc<Mutex<TempMatrixPool>>,
    optimizers: OptimizerStore,
) -> JobResult {
    let gpu_ref: Option<&crate::compute_manager::operators_v2::gpu_v2::GpuCompute> =
        job.gpu_compute.as_deref();

    let mut map = optimizers.lock().unwrap();
    let expr = map.get_mut(&job.buffer_idx).unwrap_or_else(|| {
        panic!(
            "OptimizerApplyUpdate: OptimizerExpr for buffer_idx={} not found. \
             This means `modify_grads` phase was skipped for this buffer. \
             Phase order violation (MIGRATION_PLAN.md §2, инвариант I-1).",
            job.buffer_idx
        )
    });

    expr.apply_update_step_buffered_handle_hybrid(&job.params, &job.grads, gpu_ref);

    JobResult::Unit
}