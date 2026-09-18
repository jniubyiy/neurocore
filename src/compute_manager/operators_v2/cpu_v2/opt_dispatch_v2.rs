// src/compute_manager/operators_v2/cpu_v2/opt_dispatch_v2.rs
//
// Диспетчер шага оптимизатора по одному буферу параметров.
//
// Кубики оптимизатора — CPU-код (см. `plans::optimizer_plan::cube`),
// и они всегда исполняются на CPU. Если буферы физически лежат в VRAM,
// `OptimizerExpr::step_buffered_handle_hybrid` скачивает их на CPU, шагает
// и заливает обратно. Ссылка на `GpuCompute` приходит в job'е — сам
// CPU-оператор её не хранит.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};

use crate::compute_manager::jobs_v2::{JobResult, OptimizerStepJob};
use crate::compute_manager::operators_v2::memory_v2::buffer::TempMatrixPool;
use crate::compute_manager::operators_v2::memory_v2::MemoryExecutor;
use crate::optimizer_plan::OptimizerExpr;

/// Исполняет `OptimizerStepJob`. Состояние оптимизатора хранится в
/// `optimizers` и переиспользуется между вызовами.
pub fn execute_optimizer_step(
    job: OptimizerStepJob,
    memory_executor: Arc<RwLock<MemoryExecutor>>,
    pool: Arc<Mutex<TempMatrixPool>>,
    optimizers: Arc<Mutex<HashMap<usize, OptimizerExpr>>>,
) -> JobResult {
    // Создаём OptimizerExpr при первом обращении к buffer_idx.
    {
        let mut map = optimizers.lock().unwrap();
        if !map.contains_key(&job.buffer_idx) {
            let num_params = job.params.rows();
            let chain = job.optimizer.build_chain();
            let mut p = pool.lock().unwrap();
            let expr = OptimizerExpr::new_buffered_handle(
                memory_executor,
                num_params,
                chain,
                &mut *p,
            );
            map.insert(job.buffer_idx, expr);
        }
    }

    let gpu_ref: Option<&crate::compute_manager::operators_v2::gpu_v2::GpuCompute> =
        job.gpu_compute.as_deref();

    let mut map = optimizers.lock().unwrap();
    let expr = map
        .get_mut(&job.buffer_idx)
        .expect("OptimizerExpr must exist after insert");

    expr.step_buffered_handle_hybrid(&job.params, &job.grads, gpu_ref);

    JobResult::Unit
}